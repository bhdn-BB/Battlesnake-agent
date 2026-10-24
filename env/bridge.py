import json
import os
import subprocess
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = (ROOT / "env/engine/REVISION").read_text().strip()


class Engine:
    def __init__(self, executable=None):
        path = (
            executable
            or os.getenv("BATTLESNAKE_ENGINE")
            or ROOT / "env/engine" / ("adapter.exe" if os.name == "nt" else "adapter")
        )
        self.process = subprocess.Popen(
            [str(path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.lock = threading.Lock()

    def batch(self, requests):
        with self.lock:
            self.process.stdin.write(json.dumps(requests, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"rules adapter exited: {self.process.poll()}")
            results = json.loads(line)
        if len(results) != len(requests):
            raise RuntimeError("adapter response count mismatch")
        for r in results:
            if r.get("error"):
                raise RuntimeError(r["error"])
        return results

    def call(self, op, env="0", **kwargs):
        return self.batch([dict(op=op, env=env, **kwargs)])[0]

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
