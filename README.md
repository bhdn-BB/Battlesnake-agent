# Battlesnake

## Installation and Usage

### 1. Install uv and necessary tools

Install `uv` on Linux/macOS:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For container deployment, install [Docker Engine](https://docs.docker.com/engine/install/) on Linux or [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/) on Windows. Docker is optional for local development.

### 2. Create and activate the virtual environment

From the repository root, on Linux/macOS:

```sh
uv sync
source .venv/bin/activate
```

`uv sync` installs the locked dependencies into `.venv`. Activation is optional when using `uv run`. [uv project guide](https://docs.astral.sh/uv/guides/projects/)

### 3. Run your Battlesnake

```sh
uv run python main.py
```

Open [localhost:8000](http://localhost:8000) for metadata and [localhost:8000/docs](http://localhost:8000/docs) for the interactive API reference.

See [deployment and submission instructions](docs/README.md#deploy-and-submit-the-url), the [competition overview](OVERVIEW.md), and [five RL pipelines](docs/RL_PIPELINES.md).

## Play a Game Locally

Install the [Battlesnake CLI](https://github.com/BattlesnakeOfficial/rules/tree/main/cli)
* You can [download compiled binaries here](https://github.com/BattlesnakeOfficial/rules/releases)
* or [install as a go package](https://github.com/BattlesnakeOfficial/rules/tree/main/cli#installation) (requires Go 1.18 or higher)

Command to run a local game

```sh
battlesnake play -W 11 -H 11 --name 'Python Starter Project' --url http://127.0.0.1:8000 -g solo --browser
```

