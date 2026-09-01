# ── NHTSA Early Warning System — Docker Image ─────────────────────────────
# Base: Python 3.12-slim (project tested on 3.12; matches Colab T4 environment)
#
# Build (full ML stack, ~5 GB):
#   docker build -t nhtsa-early-warning .
#
# Build (no SBERT/BART, ~1.5 GB):
#   docker build -t nhtsa-early-warning:lite --build-arg SKIP_HEAVY_ML=true .
#
# Run full pipeline:
#   docker compose up pipeline
#
# Run one phase:
#   PHASE=1 docker compose up pipeline
FROM python:3.12-slim

# gcc/g++ required if hdbscan or umap-learn lack pre-built wheels for this platform.
# Removed from final image layer via chained RUN to avoid bloat.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────
COPY requirements.txt .

ARG SKIP_HEAVY_ML=false
# Override for corporate networks where pypi.org is behind an SSL-inspection proxy.
# Example: --build-arg PIP_INDEX_URL=https://jfrog.devstack.vwgroup.com/artifactory/api/pypi/pypi/simple/
ARG PIP_INDEX_URL=https://pypi.org/simple/
# Space-separated hostnames to mark as trusted (skips cert check for that host only).
ARG PIP_TRUSTED_HOST=

# Write pip.conf so every pip invocation in this image picks up the custom index.
RUN if [ -n "$PIP_INDEX_URL" ] && [ "$PIP_INDEX_URL" != "https://pypi.org/simple/" ]; then \
        pip config set global.index-url "$PIP_INDEX_URL"; \
    fi \
 && if [ -n "$PIP_TRUSTED_HOST" ]; then \
        pip config set global.trusted-host "$PIP_TRUSTED_HOST"; \
    fi

# Full build: CPU-only torch (~400 MB) then the rest of requirements.txt.
# Lite build (SKIP_HEAVY_ML=true): lightweight deps only — no torch/transformers.
RUN if [ "$SKIP_HEAVY_ML" = "false" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
        && pip install --no-cache-dir -r requirements.txt; \
    else \
        pip install --no-cache-dir \
            pandas numpy nltk scikit-learn scipy \
            matplotlib seaborn tqdm requests pyyaml pytest; \
    fi \
 && python -c "\
import nltk; \
[nltk.download(p, download_dir='/usr/share/nltk_data', quiet=True) \
 for p in ('stopwords','punkt','wordnet','punkt_tab','averaged_perceptron_tagger_eng')]" \
 && rm -rf /root/.cache/pip /tmp/*

# ── Application source ─────────────────────────────────────────────────────
COPY config.py      .
COPY configs/       configs/
COPY scripts/       scripts/
COPY src/           src/
COPY tests/         tests/

# Data directories; overlaid by a bind-mount volume at runtime so outputs
# written inside the container are visible on the host after the run.
RUN mkdir -p data/raw data/processed data/outputs

# ── Runtime environment ────────────────────────────────────────────────────
ENV PHASE=all \
    SKIP_SEMANTIC=false \
    FORCE_REFETCH=false \
    NLTK_DATA=/usr/share/nltk_data \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/hub \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "scripts/entrypoint.py"]
