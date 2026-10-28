FROM python:3.13-slim AS runtime

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /bin/
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY battlesnake ./battlesnake
COPY training ./training
COPY env ./env
COPY LICENSES ./LICENSES
COPY THIRD_PARTY_NOTICES.md ./

ENV PATH="/app/.venv/bin:$PATH"
ENV PORT=8000 PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "battlesnake"]

FROM runtime AS learned
RUN uv sync --locked --no-dev --no-install-project --extra inference

FROM runtime AS final
