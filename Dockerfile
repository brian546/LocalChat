FROM python:3.12-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_DEV=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Build tools are needed for some wheels (for example pytrec-eval).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock /app/

# Install dependencies from lockfile for reproducible builds.
RUN uv sync --locked

COPY configs /app/configs
COPY src /app/src
COPY data /app/data

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "src/rag_agent/chatbot.py", "--server.address=0.0.0.0", "--server.port=8501"]
