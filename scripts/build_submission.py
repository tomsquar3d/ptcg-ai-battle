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


def build(version: str) -> None:
    target = os.path.join(REPO, "submissions", version)
    if not os.path.isdir(target):
        raise SystemExit(f"no such submission dir: {target}")

    # 1) sync shared code
    src_ptcg = os.path.join(REPO, "src", "ptcg")
    dst_ptcg = os.path.join(target, "ptcg")
    shutil.rmtree(dst_ptcg, ignore_errors=True)
    shutil.copytree(src_ptcg, dst_ptcg, ignore=shutil.ignore_patterns("__pycache__"))

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
