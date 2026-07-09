#!/usr/bin/env python3
"""Assemble a submission directory into a self-contained bundle.

Each submissions/vN/ must be standalone on Kaggle (main.py + deck.csv + cg/ + any shared
code), because src/ is not shipped. This vendors the canonical shared code (src/ptcg) and the
native SDK (cg/, whose compiled libs are git-ignored) into the target submission so it runs and
can be tar'd. src/ptcg is the source of truth; submissions/vN/ptcg is a synced snapshot.

Usage:  .venv/bin/python scripts/build_submission.py v1
        (then:  cd submissions/v1 && tar -czvf ../../submission.tar.gz * )
"""
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Per-version ONNX source (subdir under experiments/checkpoints). Default = the checkpoints root (v5).
_ONNX_SUBDIR = {"v6": "ladder2", "v7": "ladder3"}


def build(version: str) -> None:
    target = os.path.join(REPO, "submissions", version)
    if not os.path.isdir(target):
        raise SystemExit(f"no such submission dir: {target}")

    # 1) sync shared code (src/ptcg + src/search — the belief engine and the search layer)
    for pkg in ("ptcg", "search"):
        src_pkg = os.path.join(REPO, "src", pkg)
        if not os.path.isdir(src_pkg):
            continue
        dst_pkg = os.path.join(target, pkg)
        shutil.rmtree(dst_pkg, ignore_errors=True)
        shutil.copytree(src_pkg, dst_pkg, ignore=shutil.ignore_patterns("__pycache__"))

    # 1b) neural submissions only: vendor a TORCH-FREE nn subset (featurization + the ONNX guide)
    #     and copy the trained ONNX models in. Gated on the agent actually importing onnx_guide, so
    #     non-neural submissions stay lean and torch-free.
    agent_src = ""
    agent_py = os.path.join(target, "agent.py")
    if os.path.exists(agent_py):
        with open(agent_py) as fh:
            agent_src = fh.read()
    if "onnx_guide" in agent_src:
        dst_nn = os.path.join(target, "nn")
        shutil.rmtree(dst_nn, ignore_errors=True)
        os.makedirs(dst_nn)
        for f in ("__init__.py", "features.py", "onnx_guide.py"):   # NO encoder/guide/dataset (torch)
            shutil.copy2(os.path.join(REPO, "src", "nn", f), os.path.join(dst_nn, f))
        ckpt_dir = os.path.join(REPO, "experiments", "checkpoints", _ONNX_SUBDIR.get(version, ""))
        for model in ("value.onnx", "policy.onnx"):
            src_model = os.path.join(ckpt_dir, model)
            if os.path.exists(src_model):
                shutil.copy2(src_model, os.path.join(target, model))
            elif not os.path.exists(os.path.join(target, model)):
                raise SystemExit(f"missing ONNX model for {version}: {model} "
                                 f"(run scripts/export_onnx.py)")

    # 2) ensure the SDK (with native libs) is present — copy from v0 if missing
    dst_cg = os.path.join(target, "cg")
    if not os.path.isdir(dst_cg):
        shutil.copytree(os.path.join(REPO, "submissions", "v0", "cg"), dst_cg,
                        ignore=shutil.ignore_patterns("__pycache__"))

    # 3) sanity: required files present
    for f in ("main.py", "deck.csv"):
        if not os.path.exists(os.path.join(target, f)):
            raise SystemExit(f"missing required file: {version}/{f}")

    # clean caches
    for root, dirs, _ in os.walk(target):
        for d in list(dirs):
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)

    print(f"built {version}: synced ptcg from src, cg present. "
          f"Bundle with:  cd submissions/{version} && tar -czvf ../../submission.tar.gz *")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "v1")
