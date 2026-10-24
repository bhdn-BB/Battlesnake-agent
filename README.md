# Battlesnake

## Requirements

- Git
- Python 3.12
- Go 1.24.2, available on `PATH`

## Clone the repository

```sh
git clone https://github.com/bhdn-BB/Deep-RL-KPI.git
cd Deep-RL-KPI
```

## Install dependencies

```sh
python -m pip install uv==0.11.8
uv python install 3.12
uv sync --locked --project training --python 3.12
```

`uv` creates the virtual environment and installs the pinned dependencies.
Manual activation is not required for the commands below.

## Build the game engine

Windows PowerShell:

```powershell
go -C env/engine build -mod=readonly -o adapter.exe .
```

Linux/macOS:

```sh
go -C env/engine build -mod=readonly -o adapter .
```

## Start the web app

Run from the repository root:

```sh
uv run --locked --project training python -m webapp.server
```

Open **http://127.0.0.1:8080** in your browser. Use arrow keys or WASD to play.
Press `Ctrl+C` in the terminal to stop the app.

Trained weights are not included in the repository. Without weights, select the
scripted Teacher opponent. To play against a trained model, place its `.ts` file
and matching `.json` manifest under `runs/duel_v3/`, then refresh the model list.
