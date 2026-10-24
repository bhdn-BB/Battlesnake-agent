import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from .runtime import PolicyRuntime


@asynccontextmanager
async def lifespan(app):
    app.state.policy = PolicyRuntime()
    yield
    app.state.policy.close()


app = FastAPI(title="Battlesnake", version="1.0.0", lifespan=lifespan)


@app.get("/")
def info() -> dict:
    return {
        "apiversion": "1",
        "author": os.getenv("BATTLESNAKE_AUTHOR", ""),
        "color": "#2563eb",
        "head": "default",
        "tail": "default",
        "version": app.version,
    }


@app.post("/start")
def start(state: dict) -> dict:
    app.state.policy.start(state)
    return {}


@app.post("/end")
def end(state: dict) -> dict:
    app.state.policy.end(state)
    return {}


@app.post("/move")
def move(state: dict) -> dict:
    return {"move": app.state.policy.move(state)}


def main():
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
