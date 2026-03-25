from __future__ import annotations

from functools import lru_cache

from g2p_en import G2p


def _resource_exists(nltk_module, resource_paths: tuple[str, ...]) -> bool:
    for resource_path in resource_paths:
        try:
            nltk_module.data.find(resource_path)
        except LookupError:
            continue
        else:
            return True
    return False


def _ensure_nltk_resource(
    nltk_module,
    *,
    resource_paths: tuple[str, ...],
    download_names: tuple[str, ...],
) -> None:
    if _resource_exists(nltk_module, resource_paths):
        return

    download_errors: list[str] = []
    for download_name in download_names:
        try:
            download_succeeded = bool(nltk_module.download(download_name, quiet=True))
        except Exception as error:  # pragma: no cover - defensive guard for downloader failures
            download_errors.append(f"{download_name}: {error}")
            continue

        if download_succeeded and _resource_exists(nltk_module, resource_paths):
            return

        status = "download returned false" if not download_succeeded else "resource still missing after download"
        download_errors.append(f"{download_name}: {status}")

    attempted_downloads = ", ".join(download_errors) if download_errors else "no downloads were attempted"
    raise LookupError(
        "Required NLTK resources are unavailable. "
        f"Checked {resource_paths} and attempted {attempted_downloads}."
    )


@lru_cache(maxsize=1)
def ensure_nltk_resources() -> None:
    import nltk

    _ensure_nltk_resource(
        nltk,
        resource_paths=("corpora/cmudict",),
        download_names=("cmudict",),
    )
    _ensure_nltk_resource(
        nltk,
        resource_paths=(
            "taggers/averaged_perceptron_tagger_eng",
            "taggers/averaged_perceptron_tagger",
        ),
        download_names=(
            "averaged_perceptron_tagger_eng",
            "averaged_perceptron_tagger",
        ),
    )


@lru_cache(maxsize=1)
def get_g2p() -> G2p:
    ensure_nltk_resources()
    g2p = G2p()
    g2p("warmup")
    return g2p


@lru_cache(maxsize=256)
def phonemize_text(text: str) -> list[list[str]]:
    g2p = get_g2p()
    words = [word for word in text.strip().split() if word]
    phoneme_words: list[list[str]] = []

    for word in words:
        phonemes = [token for token in g2p(word) if token and token != " "]
        cleaned_phonemes = [
            "".join(character for character in phoneme if not character.isdigit())
            for phoneme in phonemes
        ]
        phoneme_words.append([phoneme for phoneme in cleaned_phonemes if phoneme])

    return phoneme_words


def estimate_phoneme_segments(target_text: str, word_segments: list[dict]) -> list[dict]:
    phoneme_words = phonemize_text(target_text)
    phoneme_segments: list[dict] = []

    for word_segment, phonemes in zip(word_segments, phoneme_words, strict=False):
        if not phonemes:
            continue

        word_start_time = float(word_segment["start_time"])
        word_end_time = float(word_segment["end_time"])
        word_duration = max(0.0, word_end_time - word_start_time)
        phoneme_duration = word_duration / len(phonemes) if phonemes else 0.0

        for phoneme_index, phoneme in enumerate(phonemes):
            start_time = word_start_time + (phoneme_index * phoneme_duration)
            end_time = (
                word_end_time
                if phoneme_index == len(phonemes) - 1
                else word_start_time + ((phoneme_index + 1) * phoneme_duration)
            )

            phoneme_segments.append(
                {
                    "word": word_segment["text"],
                    "phoneme": phoneme,
                    "start_time": start_time,
                    "end_time": end_time,
                    "source": "estimated_from_word_alignment",
                }
            )

    return phoneme_segments
