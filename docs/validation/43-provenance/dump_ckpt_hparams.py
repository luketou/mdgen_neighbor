"""Dump a Lightning checkpoint's hyper_parameters to JSON (run on HPC: the checkpoint stays there)."""
import argparse, hashlib, json
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

digest = hashlib.sha256()
with open(args.checkpoint, "rb") as f:
    for block in iter(lambda: f.read(1 << 20), b""):
        digest.update(block)

ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
hparams = ckpt.get("hyper_parameters", {})
if "args" in hparams and not isinstance(hparams["args"], dict):
    hparams = {**hparams, "args": vars(hparams["args"])}

report = {
    "checkpoint": str(args.checkpoint),
    "sha256": digest.hexdigest(),
    "top_level_keys": sorted(ckpt),
    "epoch": ckpt.get("epoch"),
    "global_step": ckpt.get("global_step"),
    "pytorch_lightning_version": ckpt.get("pytorch-lightning_version"),
    "hyper_parameters": hparams,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=repr) + "\n", encoding="utf-8", newline="\n")
print(json.dumps({k: report[k] for k in ("sha256", "epoch", "global_step")}))
print("abs_pos_emb =", report["hyper_parameters"].get("args", {}).get("abs_pos_emb", "<missing>"))
