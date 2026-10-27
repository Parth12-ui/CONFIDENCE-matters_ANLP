#!/usr/bin/env python3
"""
Rebuild meta-llama/Meta-Llama-3.1-8B-Instruct's tokenizer_config.json without a token.

The ungated NousResearch/Meta-Llama-3.1-8B-Instruct copy matches Meta's repo file for
file except tokenizer_config.json, which carries an older chat template. unsloth's copy
has Meta's current chat template but adds padding settings. Removing unsloth's five added
keys and restoring tokenizer_class gives a file whose git blob hash equals Meta's
(db88166e...), i.e. byte-identical. The script checks that hash and refuses to write
anything else.

Usage (then copy the output over the NousResearch download and run verify_model.py):
  python3 scripts/llama_tokenizer.py OUTPUT_PATH
"""

import hashlib
import re
import sys
import urllib.request

SOURCE = "https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct/resolve/main/tokenizer_config.json"
META_BLOB = "db88166e2bc4c799fd5d1ae643b75e84d03ee70e"
ADDED_KEYS = ("add_bos_token", "extra_special_tokens", "pad_token", "padding_side", "unk_token")


def main():
    out = sys.argv[1]
    with urllib.request.urlopen(SOURCE, timeout=60) as r:
        lines = r.read().decode().split("\n")
    keep = [l for l in lines if not any(re.match(rf'\s*"{k}"\s*:', l) for k in ADDED_KEYS)]
    text = "\n".join(keep)
    text = text.replace('"tokenizer_class": "PreTrainedTokenizer",', '"tokenizer_class": "PreTrainedTokenizerFast"')
    text = re.sub(r",(\n})", r"\1", text)
    data = text.encode()
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    if blob != META_BLOB:
        sys.exit(f"Hash {blob} does not match Meta's {META_BLOB}; unsloth's file may have changed. Nothing written.")
    with open(out, "wb") as f:
        f.write(data)
    print(f"Wrote {out} (git blob {blob}, identical to Meta's tokenizer_config.json)")


if __name__ == "__main__":
    main()
