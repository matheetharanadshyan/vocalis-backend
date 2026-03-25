FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/usr/local/share/nltk_data

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        espeak-ng \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN mkdir -p "${NLTK_DATA}" && python - <<'PY'
import nltk

download_dir = "/usr/local/share/nltk_data"

def resource_exists(*resource_paths: str) -> bool:
    for resource_path in resource_paths:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            continue
        else:
            return True
    return False


if not nltk.download("cmudict", download_dir=download_dir, quiet=True):
    raise SystemExit("Failed to download NLTK resource: cmudict")

for resource_name in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
    nltk.download(resource_name, download_dir=download_dir, quiet=True)

if not resource_exists("corpora/cmudict"):
    raise SystemExit("Downloaded cmudict but it is still unavailable")

if not resource_exists(
    "taggers/averaged_perceptron_tagger_eng",
    "taggers/averaged_perceptron_tagger",
):
    raise SystemExit("No compatible averaged perceptron tagger was downloaded")
PY

COPY . .

EXPOSE 8000

CMD ["/bin/sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
