#!/usr/bin/env python3
"""
Hidden states for the logged conversations, for the truth probes in
src/analysis/hidden_probe.py. No new text is generated: each conversation is rebuilt from
turns.jsonl (system prompt, then each turn's logged prompt and the response the model saw
in its history, including any loop trimming or completed verdict) and read in a single
forward pass. In a causal model the state at a position depends only on the tokens before
it, so one pass over the whole conversation gives every turn's states.

For every turn t we keep the residual stream (output of each decoder layer, plus the
embeddings as layer 0) at two positions:
  - pre:  last token of the generation prompt for turn t, after the user's turn-t message
          and before the model's turn-t response;
  - post: last token of the model's turn-t answer as the chat template closes it.

Output (one file per track): <out_dir>/<track>.npz with
  states   float16 [conversations, turns, 2 (pre, post), layers kept, hidden size]
  claim_id int     [conversations]
  layers   int     [layers kept]
and <out_dir>/<track>_meta.json with the run settings.

Usage (from the repository root, on a GPU):
  python3 src/extract_hidden.py --model_name Qwen/Qwen2.5-7B-Instruct \
      --results_dir results/qwen_math --tracks control negative --load_in_4bit
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiment import REASONING_FIRST_SYSTEM_PROMPT  # noqa: E402

POSITIONS = ("pre", "post")


def load_conversations(path, limit=None):
    """{claim_id: [record for turn 0..T]} in turn order, complete conversations only."""
    convs = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            convs.setdefault(r["claim_id"], {})[r["turn"]] = r
    out = {}
    for cid in sorted(convs):
        turns = convs[cid]
        if sorted(turns) == list(range(len(turns))):
            out[cid] = [turns[t] for t in range(len(turns))]
    if limit:
        out = dict(list(out.items())[:limit])
    return out


def positions(tokenizer, records, template_kwargs):
    """Token ids of the whole conversation and, per turn, the (pre, post) token positions."""
    msgs = [{"role": "system", "content": REASONING_FIRST_SYSTEM_PROMPT}]
    marks = []
    for r in records:
        msgs.append({"role": "user", "content": r["prompt"]})
        pre = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **template_kwargs)
        msgs.append({"role": "assistant", "content": r["response"]})
        post = tokenizer.apply_chat_template(msgs, tokenize=False, **template_kwargs)
        marks.append((pre, post))
    full = marks[-1][1]
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    starts = np.array([s for s, _ in enc["offset_mapping"]])
    pos = []
    for pre, post in marks:
        # The prefix ends inside the full text; its position is the last token starting
        # before that point (offsets rather than re-tokenising the prefix, since BPE can
        # merge across the boundary, e.g. a newline that starts the response)
        row = []
        for text in (pre, post):
            assert full.startswith(text), "chat template is not prefix-stable"
            row.append(int(np.searchsorted(starts, len(text), side="left")) - 1)
        pos.append(row)
    return torch.tensor(enc["input_ids"]), pos


def load_model(name, load_in_4bit):
    native_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if native_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
    kw = {"torch_dtype": dtype}
    if torch.cuda.is_available():
        kw["device_map"] = "auto"
    if load_in_4bit:
        # Same quantisation as the experiment runs (src/run_experiment.py)
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    model.eval()
    return model


@torch.no_grad()
def states_at(model, ids, pos, layers):
    """[turns, 2, len(layers), hidden] at the given positions, layer 0 being the embeddings."""
    flat = torch.tensor([p for row in pos for p in row])
    keep = {}
    blocks = model.model.layers

    def hook(i):
        def f(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            keep[i] = h[0, flat.to(h.device)].float().cpu()
        return f

    handles = [blocks[i - 1].register_forward_hook(hook(i)) for i in layers if i > 0]
    if 0 in layers:
        handles.append(model.model.embed_tokens.register_forward_hook(
            lambda _m, _i, out: keep.__setitem__(0, out[0, flat.to(out.device)].float().cpu())))
    try:
        # Only the decoder stack is needed; skipping the LM head avoids a sequence x vocabulary tensor
        model.model(input_ids=ids[None].to(model.device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    x = torch.stack([keep[i] for i in layers], dim=1)  # [turns*2, layers, hidden]
    return x.view(len(pos), 2, len(layers), -1).numpy().astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--tracks", nargs="+", default=["control", "negative"])
    ap.add_argument("--out_dir", default=None, help="default: <results_dir>/hidden")
    ap.add_argument("--layer_stride", type=int, default=2, help="keep every n-th layer (and the last)")
    ap.add_argument("--limit", type=int, default=None, help="first N conversations per track (tests)")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--date_string", default=None,
                    help="Llama 3.1 only: date its chat template writes into the system header")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.results_dir, "hidden")
    os.makedirs(out_dir, exist_ok=True)
    template_kwargs = {"date_string": args.date_string} if args.date_string else {}

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = load_model(args.model_name, args.load_in_4bit)
    n_layers = model.config.num_hidden_layers
    layers = sorted(set(range(0, n_layers + 1, args.layer_stride)) | {n_layers})
    print(f"{args.model_name}: {n_layers} layers, keeping {layers}")

    for track in args.tracks:
        convs = load_conversations(os.path.join(args.results_dir, track, "turns.jsonl"), args.limit)
        out = os.path.join(out_dir, f"{track}.npz")
        all_states, cids, lengths = [], [], []
        for k, (cid, recs) in enumerate(convs.items()):
            ids, pos = positions(tok, recs, template_kwargs)
            all_states.append(states_at(model, ids, pos, layers))
            cids.append(cid)
            lengths.append(len(ids))
            if k % 25 == 0:
                peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
                print(f"  {track}: {k + 1}/{len(convs)} conversations, {len(ids)} tokens, peak {peak:.1f} GB", flush=True)
        np.savez(out, states=np.stack(all_states), claim_id=np.array(cids), layers=np.array(layers))
        json.dump({"model": args.model_name, "results_dir": args.results_dir, "track": track,
                   "conversations": len(cids), "turns": all_states[0].shape[0], "positions": POSITIONS,
                   "layers": layers, "load_in_4bit": args.load_in_4bit, "date_string": args.date_string,
                   "max_tokens": int(max(lengths))},
                  open(os.path.join(out_dir, f"{track}_meta.json"), "w"), indent=1)
        print(f"wrote {out}: {np.stack(all_states).shape}")


if __name__ == "__main__":
    main()
