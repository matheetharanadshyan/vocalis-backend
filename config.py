import torch
from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings


def parse_csv_setting(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    huggingface_model_id: str = Field(default="matheetharanadshyan/wav2vec2-svarah")
    phoneme_model_id: str = Field(default="facebook/wav2vec2-lv-60-espeak-cv-ft")
    use_quantized_ctc_model: bool = Field(default=True)
    use_quantized_phoneme_model: bool = Field(default=True)
    use_phoneme_model: bool = Field(default=True)
    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    use_groq: bool = Field(default=True)
    device: str = Field(default_factory=get_device)
    debug: bool = Field(default=False)
    app_name: str = Field(default="Vocalis")
    turso_database_url: str = Field(default="")
    turso_auth_token: str = Field(default="")
    max_file_size_mb: int = Field(default=50)
    request_timeout_seconds: int = Field(default=300)
    sample_rate: int = Field(default=16000)
    default_websocket_sample_rate: int = Field(default=48000)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    cors_allowed_origins: str = Field(default="http://localhost:5173,http://127.0.0.1:5173")
    trim_top_db: int = Field(default=25)
    max_audio_duration_seconds: float = Field(default=15.0)
    min_audio_duration_seconds: float = Field(default=0.3)
    max_websocket_buffer_mb: int = Field(default=12)
    websocket_chunk_timeout_seconds: int = Field(default=30)
    default_audio_channels: int = Field(default=1)
    default_sample_width_bytes: int = Field(default=2)
    default_audio_encoding: str = Field(default="pcm_s16le")
    log_level: str = Field(default="INFO")
    http_requests_per_minute: int = Field(default=60)
    websocket_messages_per_minute: int = Field(default=240)
    io_worker_threads: int = Field(default=8)
    model_worker_threads: int = Field(default=4)
    max_concurrent_alignment_tasks: int = Field(default=2)
    max_concurrent_phoneme_tasks: int = Field(default=2)
    max_concurrent_feedback_tasks: int = Field(default=8)
    groq_timeout_seconds: int = Field(default=20)
    groq_retry_attempts: int = Field(default=2)
    groq_retry_backoff_seconds: float = Field(default=1.0)
    groq_circuit_breaker_threshold: int = Field(default=3)
    groq_circuit_breaker_seconds: int = Field(default=60)
    auth_session_days_valid: int = Field(default=30)
    auth_rotate_session_on_me: bool = Field(default=True)
    auth_rate_limit_attempts: int = Field(default=5)
    auth_rate_limit_window_seconds: int = Field(default=300)

    @property
    def cors_allowed_origins_list(self) -> list[str]:
        return parse_csv_setting(self.cors_allowed_origins)


settings = Settings()
