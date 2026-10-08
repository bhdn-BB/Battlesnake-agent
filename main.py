import os

import uvicorn
from fastapi import FastAPI

from heuristic_policy import choose_move

app = FastAPI(title="Battlesnake", version="1.0.0")


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
@app.post("/end")
def lifecycle(state: dict) -> dict:
    return {}


@app.post("/move")
def move(state: dict) -> dict:
    return {"move": choose_move(state)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
