"""Artifact hashes, atomic manifests and policy copies; standard library only."""

import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def copy_policy(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".ts", ".json"):
        temporary = target.with_suffix(suffix + ".tmp")
        shutil.copy2(source.with_suffix(suffix), temporary)
        os.replace(temporary, target.with_suffix(suffix))
    return target
