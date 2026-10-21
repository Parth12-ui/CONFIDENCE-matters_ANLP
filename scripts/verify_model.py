#!/usr/bin/env python3
"""
Check that a local model folder is byte-identical to a Hugging Face repo, using the
hashes the Hub publishes even for gated repos (no token needed): SHA-256 for LFS
files (weights, tokenizer.json) and the git blob hash for small files.

Used to confirm that Llama-3.1-8B-Instruct assembled from an ungated copy (weights,
config and tokenizer from NousResearch, tokenizer_config.json reconstructed from
unsloth's) equals meta-llama/Meta-Llama-3.1-8B-Instruct. Standard library only, so
it runs with the Ada login node's Python 3.9.

Usage:
  python3 scripts/verify_model.py /share1/$USER/models/Meta-Llama-3.1-8B-Instruct \
      meta-llama/Meta-Llama-3.1-8B-Instruct
"""

import hashlib
import json
import os
import sys
import urllib.request

SKIP = {"README.md", "LICENSE", "USE_POLICY.md", ".gitattributes"}


def git_blob(path: str) -> str:
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    local, repo = sys.argv[1], sys.argv[2]
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=60) as r:
        siblings = json.load(r)["siblings"]
    bad = 0
    checked = 0
    for s in siblings:
        name = s["rfilename"]
        if name in SKIP or name.startswith("original/"):
            continue
        path = os.path.join(local, name)
        if not os.path.exists(path):
            print(f"MISSING  {name}")
            bad += 1
            continue
        lfs = s.get("lfs")
        ok = sha256(path) == lfs["sha256"] if lfs else git_blob(path) == s["blobId"]
        checked += 1
        print(f"{'ok      ' if ok else 'MISMATCH'} {name}")
        bad += not ok
    print(f"{checked} files checked against {repo}: {'IDENTICAL' if bad == 0 else f'{bad} problem(s)'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
