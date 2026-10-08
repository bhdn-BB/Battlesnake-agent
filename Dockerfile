FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /bin/
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY main.py heuristic_policy.py ./

ENV PATH="/app/.venv/bin:$PATH"
ENV PORT=8000 PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "main.py"]
