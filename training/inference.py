import json
from pathlib import Path

import torch

from env.encoding import SHAPE, VERSION, action_mask, encode


def export_policy(model, path, kind, metadata=None):
    if kind != "imitation":
        raise ValueError("only the imitation → PPO policy can be exported")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    scripted = torch.jit.trace(
        model, (torch.zeros((1, *SHAPE), dtype=torch.uint8), torch.zeros(1, 128))
    )
    scripted.save(str(path))
    from env.bridge import REVISION

    from .artifacts import sha256

    manifest = dict(
        format=1,
        encoder=VERSION,
        kind=kind,
        match_snakes=2,
        actions=["up", "down", "left", "right"],
        parameters=sum(p.numel() for p in model.parameters()),
        engine_revision=REVISION,
        torch_version=torch.__version__,
        sha256=sha256(path),
    )
    manifest.update(metadata or {})
    if manifest["parameters"] >= 1_000_000:
        raise ValueError("model exceeds parameter budget")
    path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return path


class FrozenPolicy:
    def __init__(self, path, threads=1):
        torch.set_num_threads(threads)
        self.path = Path(path)
        self.metadata = json.loads(self.path.with_suffix(".json").read_text())
        from env.bridge import REVISION

        from .artifacts import sha256

        if self.metadata["engine_revision"] != REVISION or self.metadata[
            "sha256"
        ] != sha256(path):
            raise ValueError("export engine revision or SHA256 mismatch")
        if self.metadata["encoder"] != VERSION or self.metadata["actions"] != [
            "up",
            "down",
            "left",
            "right",
        ]:
            raise ValueError("incompatible exported policy")
        self.model = torch.jit.load(str(path), map_location="cpu").eval()
        if (
            self.metadata.get("kind") != "imitation"
            or self.metadata.get("match_snakes") != 2
        ):
            raise ValueError("only 1v1 imitation → PPO exports are supported")

    @torch.inference_mode()
    def act(self, board, sid, order=None, hidden=None):
        obs = torch.from_numpy(encode(board, sid, order)[None])
        hidden = (
            torch.zeros(1, 128)
            if hidden is None
            else torch.as_tensor(hidden, dtype=torch.float32).reshape(1, 128)
        )
        scores, _, nxt = self.model(obs, hidden)
        mask, _ = action_mask(board, sid)
        action = int(scores[0].masked_fill(~torch.from_numpy(mask), -1e9).argmax())
        return action, nxt.numpy().copy()
