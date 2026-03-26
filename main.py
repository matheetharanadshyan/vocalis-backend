import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from time import perf_counter

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from alignment import load_alignment_bundle
from assessment_service import assess_pronunciation, preload_assessment_dependencies
from auth import (
    create_session_record,
    enforce_sign_in_rate_limit,
    hash_password,
    hash_session_token,
    record_failed_sign_in,
    reset_sign_in_rate_limit,
    validate_password,
    validate_username,
    verify_password,
)
from auth_repository import (
    create_session,
    create_user,
    delete_session,
    delete_expired_sessions,
    get_user_by_session_token_hash,
    get_user_by_username,
    replace_session,
    update_session_last_used,
    update_user_last_seen,
)
from audio_processing import preprocess_audio_bytes
from config import settings
from database import init_database, ping_database
from personalization_repository import (
    build_user_personalization_summary,
    get_attempt_history,
)
from personalization_service import persist_personalization_state, resolve_practice_context
from target_texts import TargetTextManager

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title=settings.app_name)
logger = logging.getLogger("vocalis.backend")
io_executor = ThreadPoolExecutor(
    max_workers=settings.io_worker_threads,
    thread_name_prefix="vocalis-io",
)
model_executor = ThreadPoolExecutor(
    max_workers=settings.model_worker_threads,
    thread_name_prefix="vocalis-model",
)
alignment_semaphore = asyncio.Semaphore(settings.max_concurrent_alignment_tasks)
phoneme_model_semaphore = asyncio.Semaphore(settings.max_concurrent_phoneme_tasks)
feedback_semaphore = asyncio.Semaphore(settings.max_concurrent_feedback_tasks)
runtime_dependencies = {
    "database": {
        "label": "Database",
        "critical": True,
        "enabled": True,
        "ready": False,
        "error": "Not checked yet.",
        "checked_at": None,
    },
    "alignment_bundle": {
        "label": "Alignment bundle",
        "critical": True,
        "enabled": True,
        "ready": False,
        "error": "Not checked yet.",
        "checked_at": None,
    },
    "g2p": {
        "label": "G2P resources",
        "critical": True,
        "enabled": True,
        "ready": False,
        "error": "Not checked yet.",
        "checked_at": None,
    },
    "phoneme_model": {
        "label": "Phoneme model",
        "critical": False,
        "enabled": settings.use_phoneme_model,
        "ready": not settings.use_phoneme_model,
        "error": None if settings.use_phoneme_model else "Disabled by configuration.",
        "checked_at": None,
    },
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SignUpRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=128)


class SignInRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=128)


def mark_runtime_dependency(
    name: str,
    *,
    ready: bool,
    error: str | None = None,
) -> None:
    dependency = runtime_dependencies[name]
    dependency["ready"] = ready
    dependency["error"] = None if ready else error
    dependency["checked_at"] = datetime.now(timezone.utc).isoformat()


def build_runtime_health_payload() -> dict:
    dependencies = {
        name: {
            "label": state["label"],
            "enabled": state["enabled"],
            "critical": state["critical"],
            "ready": state["ready"],
            "error": state["error"],
            "checked_at": state["checked_at"],
        }
        for name, state in runtime_dependencies.items()
    }
    critical_failures = [
        name
        for name, state in dependencies.items()
        if state["enabled"] and state["critical"] and not state["ready"]
    ]
    optional_failures = [
        name
        for name, state in dependencies.items()
        if state["enabled"] and not state["critical"] and not state["ready"]
    ]

    if critical_failures:
        status = "unavailable"
    elif optional_failures:
        status = "degraded"
    else:
        status = "ok"

    return {
        "status": status,
        "service": settings.app_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dependencies": dependencies,
    }


async def refresh_runtime_health() -> None:
    try:
        await run_blocking(ping_database, executor=io_executor)
        mark_runtime_dependency("database", ready=True)
    except Exception as error:
        mark_runtime_dependency("database", ready=False, error=str(error))


def normalize_username(username: str) -> str:
    try:
        return validate_username(username)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def build_auth_payload(user_row, token: str | None = None) -> dict:
    payload = {
        "user": {
            "id": int(user_row[0]),
            "username": str(user_row[1]),
            "created_at": str(user_row[2]),
            "updated_at": str(user_row[3]),
            "last_seen_at": user_row[4],
        }
    }

    if token is not None:
        payload["token"] = token

    return payload


def get_current_user_from_token(authorization: str | None):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is required.")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Invalid authorization format.")

    token_hash = hash_session_token(token)
    user_session_row = get_user_by_session_token_hash(token_hash)
    if user_session_row is None:
        raise HTTPException(status_code=401, detail="Session is invalid.")

    expires_at = datetime.fromisoformat(str(user_session_row[6]))
    if expires_at <= datetime.now(timezone.utc):
        delete_session(token_hash)
        raise HTTPException(status_code=401, detail="Session has expired.")

    update_session_last_used(token_hash)
    update_user_last_seen(int(user_session_row[0]))
    return user_session_row, token_hash


def get_current_user_from_raw_token(token: str | None):
    if not token:
        return None

    token_hash = hash_session_token(token)
    user_session_row = get_user_by_session_token_hash(token_hash)
    if user_session_row is None:
        return None

    expires_at = datetime.fromisoformat(str(user_session_row[6]))
    if expires_at <= datetime.now(timezone.utc):
        delete_session(token_hash)
        return None

    update_session_last_used(token_hash)
    update_user_last_seen(int(user_session_row[0]))
    return user_session_row


async def run_blocking(
    func,
    *args,
    executor: ThreadPoolExecutor | None = None,
    semaphore: asyncio.Semaphore | None = None,
    **kwargs,
):
    loop = asyncio.get_running_loop()
    call = partial(func, *args, **kwargs)

    if semaphore is not None:
        async with semaphore:
            return await loop.run_in_executor(executor, call)

    return await loop.run_in_executor(executor, call)


async def run_timed_blocking(
    stage_name: str,
    func,
    *args,
    executor: ThreadPoolExecutor | None = None,
    semaphore: asyncio.Semaphore | None = None,
    **kwargs,
):
    started_at = perf_counter()
    try:
        return await run_blocking(
            func,
            *args,
            executor=executor,
            semaphore=semaphore,
            **kwargs,
        )
    finally:
        logger.info("%s finished in %.3fs", stage_name, perf_counter() - started_at)


async def run_timed_async(
    stage_name: str,
    awaitable,
    *,
    semaphore: asyncio.Semaphore | None = None,
):
    started_at = perf_counter()
    try:
        if semaphore is not None:
            async with semaphore:
                return await awaitable
        return await awaitable
    finally:
        logger.info("%s finished in %.3fs", stage_name, perf_counter() - started_at)


@app.on_event("startup")
async def preload_runtime_dependencies() -> None:
    await safe_preload_runtime_dependencies()
    try:
        await run_timed_blocking("database.init", init_database, executor=io_executor)
        mark_runtime_dependency("database", ready=True)
    except Exception as error:
        mark_runtime_dependency("database", ready=False, error=str(error))
        logger.exception("Failed to initialize database connection at startup.")
    try:
        await run_timed_blocking("sessions.cleanup", delete_expired_sessions, executor=io_executor)
    except Exception:
        logger.exception("Failed to cleanup expired sessions at startup.")


@app.on_event("shutdown")
async def shutdown_runtime_dependencies() -> None:
    io_executor.shutdown(wait=False, cancel_futures=True)
    model_executor.shutdown(wait=False, cancel_futures=True)


@app.get("/health")
async def health() -> dict:
    await refresh_runtime_health()
    return build_runtime_health_payload()


@app.get("/health/ready")
async def health_ready() -> JSONResponse:
    await refresh_runtime_health()
    payload = build_runtime_health_payload()
    status_code = 200 if payload["status"] != "unavailable" else 503
    return JSONResponse(status_code=status_code, content=payload)


@app.post("/auth/sign-up")
async def sign_up(payload: SignUpRequest) -> dict:
    username = normalize_username(payload.username)
    try:
        validate_password(payload.password)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    existing_user = await asyncio.to_thread(get_user_by_username, username)
    if existing_user is not None:
        raise HTTPException(status_code=409, detail="Username is already in use.")

    password_hash = hash_password(payload.password)
    user_id = await asyncio.to_thread(create_user, username, password_hash)
    session_record = create_session_record(days_valid=settings.auth_session_days_valid)
    await asyncio.to_thread(create_session, user_id, session_record.token_hash, session_record.expires_at)

    created_user = await asyncio.to_thread(get_user_by_username, username)
    if created_user is None:
        raise HTTPException(status_code=500, detail="User could not be created.")

    return build_auth_payload(created_user, token=session_record.token)


@app.post("/auth/sign-in")
async def sign_in(payload: SignInRequest) -> dict:
    username = normalize_username(payload.username)
    await asyncio.to_thread(delete_expired_sessions)

    try:
        enforce_sign_in_rate_limit(username)
    except ValueError as error:
        raise HTTPException(status_code=429, detail=str(error)) from error

    user_row = await asyncio.to_thread(get_user_by_username, username)

    if user_row is None or not verify_password(payload.password, str(user_row[2])):
        record_failed_sign_in(username)
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    reset_sign_in_rate_limit(username)
    session_record = create_session_record(days_valid=settings.auth_session_days_valid)
    await asyncio.to_thread(create_session, int(user_row[0]), session_record.token_hash, session_record.expires_at)
    await asyncio.to_thread(update_user_last_seen, int(user_row[0]))
    refreshed_user = await asyncio.to_thread(get_user_by_username, username)

    return build_auth_payload(refreshed_user or user_row, token=session_record.token)


@app.post("/auth/sign-out")
async def sign_out(authorization: str | None = Header(default=None)) -> dict:
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is required.")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Invalid authorization format.")

    try:
        await asyncio.to_thread(delete_session, hash_session_token(token))
    except Exception:
        logger.exception("Failed to delete session during sign out.")
    return {"status": "signed_out"}


@app.get("/auth/me")
async def me(authorization: str | None = Header(default=None)) -> dict:
    await asyncio.to_thread(delete_expired_sessions)
    user_session_row, token_hash = await asyncio.to_thread(get_current_user_from_token, authorization)

    response_payload = {
        "user": {
            "id": int(user_session_row[0]),
            "username": str(user_session_row[1]),
            "created_at": str(user_session_row[2]),
            "updated_at": str(user_session_row[3]),
            "last_seen_at": user_session_row[4],
        }
    }

    if settings.auth_rotate_session_on_me:
        session_record = create_session_record(days_valid=settings.auth_session_days_valid)
        try:
            await asyncio.to_thread(
                replace_session,
                token_hash,
                int(user_session_row[0]),
                session_record.token_hash,
                session_record.expires_at,
            )
        except Exception:
            logger.exception("Failed to rotate session token during /auth/me; continuing with the existing token.")
        else:
            response_payload["token"] = session_record.token

    return response_payload


@app.get("/profile/summary")
async def profile_summary(authorization: str | None = Header(default=None)) -> dict:
    user_session_row, _token_hash = await asyncio.to_thread(get_current_user_from_token, authorization)
    summary = await asyncio.to_thread(build_user_personalization_summary, int(user_session_row[0]))
    return summary


@app.get("/attempts")
async def attempts(authorization: str | None = Header(default=None)) -> dict:
    user_session_row, _token_hash = await asyncio.to_thread(get_current_user_from_token, authorization)
    history_rows = await asyncio.to_thread(get_attempt_history, int(user_session_row[0]))
    return {
        "attempts": [
            {
                "id": int(row[0]),
                "target_text": str(row[1]),
                "target_difficulty": str(row[2]),
                "overall_score": float(row[3]),
                "performance_band": str(row[4]),
                "feedback_summary": str(row[5]),
                "created_at": str(row[6]),
            }
            for row in history_rows
        ]
    }


async def safe_send_json(websocket: WebSocket, payload: dict) -> bool:
    try:
        await websocket.send_json(payload)
        return True
    except WebSocketDisconnect:
        return False
    except RuntimeError:
        return False


async def safe_preload_runtime_dependencies() -> None:
    await preload_assessment_dependencies(
        run_timed_blocking=run_timed_blocking,
        model_executor=model_executor,
        load_alignment_bundle=load_alignment_bundle,
        logger=logger,
        on_dependency_status=mark_runtime_dependency,
        use_phoneme_model=settings.use_phoneme_model,
    )


@app.websocket("/ws/pronunciation")
async def pronunciation_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    target_text_manager = TargetTextManager()
    token = websocket.query_params.get("token")
    authenticated_user = await run_blocking(
        get_current_user_from_raw_token,
        token,
        executor=io_executor,
    )
    user_id = int(authenticated_user[0]) if authenticated_user is not None else None
    personalization_summary, focus_phonemes, current_target_text, current_target_focus_matches = await resolve_practice_context(
        user_id=user_id,
        target_text_manager=target_text_manager,
        run_timed_blocking=run_timed_blocking,
        io_executor=io_executor,
    )

    sent = await safe_send_json(
        websocket,
        {
            "type": "connection.ready",
            "message": "WebSocket Connection Established.",
            "authenticated": user_id is not None,
        },
    )
    if not sent:
        return

    sent = await safe_send_json(
        websocket,
        {
            "type": "target.assigned",
            "message": "Initial Target Text Assigned.",
            "reason": "initial",
            "target_text": current_target_text["text"],
            "target_difficulty": current_target_text["difficulty"],
            "selection_reason": (
                f"Chosen to help with /{current_target_focus_matches[0]}/."
                if current_target_focus_matches
                else None
            ),
        },
    )
    if not sent:
        return

    try:
        while True:
            audio_bytes = await websocket.receive_bytes()

            sent = await safe_send_json(
                websocket,
                {
                    "type": "recording.received",
                    "message": "Audio Received By The Backend.",
                    "bytes_received": len(audio_bytes),
                },
            )
            if not sent:
                return

            sent = await safe_send_json(
                websocket,
                {
                    "type": "audio.processing",
                    "message": "Backend Preprocessing Audio...",
                    "stage": "preprocessing",
                },
            )
            if not sent:
                return

            async def notify_assessment_progress(payload: dict) -> None:
                stage_sent = await safe_send_json(websocket, payload)
                if not stage_sent:
                    raise WebSocketDisconnect

            assessment = await assess_pronunciation(
                audio_bytes=audio_bytes,
                current_target_text=current_target_text,
                preprocess_audio_bytes=preprocess_audio_bytes,
                run_timed_blocking=run_timed_blocking,
                run_timed_async=run_timed_async,
                io_executor=io_executor,
                model_executor=model_executor,
                alignment_semaphore=alignment_semaphore,
                phoneme_model_semaphore=phoneme_model_semaphore,
                feedback_semaphore=feedback_semaphore,
                progress_notifier=notify_assessment_progress,
            )
            processed = assessment["processed"]
            alignment = assessment["alignment"]
            phoneme_segments = assessment["phoneme_segments"]
            phoneme_model_payload = assessment["phoneme_model_payload"]
            scoring_payload = assessment["scoring_payload"]
            feedback_payload = assessment["feedback_payload"]
            timings = assessment["timings"]

            sent = await safe_send_json(
                websocket,
                {
                    "type": "assessment.completed",
                    "message": "Backend Finished Processing.",
                    "stage": "completed",
                    "bytes_received": len(audio_bytes),
                    "original_sample_rate": processed["original_sample_rate"],
                    "processed_sample_rate": processed["processed_sample_rate"],
                    "original_channels": processed["original_channels"],
                    "processed_channels": processed["processed_channels"],
                    "num_samples": processed["num_samples"],
                    "duration_seconds": processed["duration_seconds"],
                    "target_text": current_target_text["text"],
                    "target_difficulty": current_target_text["difficulty"],
                    "normalized_target_text": alignment["normalized_target_text"],
                    "character_segments": alignment["character_segments"],
                    "word_segments": alignment["word_segments"],
                    "phoneme_segments": phoneme_segments,
                    "phoneme_results": scoring_payload["phoneme_results"],
                    "word_scores": scoring_payload["word_scores"],
                    "overall_score": scoring_payload["overall_score"],
                    "performance_band": scoring_payload["performance_band"],
                    "feedback_summary": feedback_payload["summary"],
                    "feedback_action_items": feedback_payload["action_items"],
                    "feedback_encouragement": feedback_payload["encouragement"],
                    "feedback_provider": feedback_payload["feedback_provider"],
                    "feedback_model": feedback_payload["feedback_model"],
                    "phoneme_model_used": phoneme_model_payload["phoneme_model_used"],
                    "phoneme_model_id": phoneme_model_payload["phoneme_model_id"],
                    "phoneme_model_transcript": phoneme_model_payload["phoneme_model_transcript"],
                    "phoneme_model_segments": phoneme_model_payload["phoneme_model_segments"],
                    "phoneme_model_error": phoneme_model_payload["phoneme_model_error"],
                    "model_quantized": alignment["model_quantized"],
                    "model_device": alignment["model_device"],
                    "processing_timings": timings,
                },
            )
            if not sent:
                return

            personalization_summary, focus_phonemes, next_target_text = await persist_personalization_state(
                user_id=user_id,
                current_target_text=current_target_text,
                alignment=alignment,
                scoring_payload=scoring_payload,
                feedback_payload=feedback_payload,
                phoneme_model_payload=phoneme_model_payload,
                target_text_manager=target_text_manager,
                run_timed_blocking=run_timed_blocking,
                io_executor=io_executor,
            )
            if personalization_summary is not None:
                sent = await safe_send_json(
                    websocket,
                    {
                        "type": "personalization.updated",
                        "message": "Personalization summary updated.",
                        "personalization_summary": personalization_summary,
                    },
                )
                if not sent:
                    return

            next_target_focus_matches = target_text_manager.match_focus_phonemes(
                next_target_text["text"],
                focus_phonemes,
            )
            current_target_text = next_target_text

            sent = await safe_send_json(
                websocket,
                {
                    "type": "target.assigned",
                    "message": "Next Target Text Assigned.",
                    "reason": "next",
                    "target_text": current_target_text["text"],
                    "target_difficulty": current_target_text["difficulty"],
                    "selection_reason": (
                        f"Chosen to help with /{next_target_focus_matches[0]}/."
                        if next_target_focus_matches
                        else None
                    ),
                },
            )
            if not sent:
                return

    except WebSocketDisconnect:
        return
    except Exception as error:
        logger.exception("Pronunciation websocket processing failed.")
        await safe_send_json(
            websocket,
            {
                "type": "audio.error",
                "message": str(error),
            },
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
