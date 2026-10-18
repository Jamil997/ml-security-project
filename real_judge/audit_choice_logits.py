"""Recover unrounded two-label logits to resolve position-averaged ties.

This numerical audit preserves all original generations and score records.
It does not optimize attacks or change prompts, weights, or audit budgets.
"""
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

ROOT = Path(__file__).resolve().parent


def main():
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    records = list(map(json.loads, (ROOT / 'prompts.jsonl').read_text().splitlines()))
    output = ROOT / 'choice_logits.jsonl'
    done = set()
    if output.exists():
        done = {r['id'] for r in map(json.loads, output.read_text().splitlines())}
    model, tokenizer = load(str(ROOT / 'model'))
    start = time.perf_counter()
    with output.open('a', buffering=1) as f:
        for i, r in enumerate(records):
            if r['id'] in done:
                continue
            prefix = r['prompt'] + protocol['forced_prefix']
            ids = tokenizer.encode(prefix, add_special_tokens=False)
            aa = tokenizer.encode(prefix + 'a', add_special_tokens=False)
            bb = tokenizer.encode(prefix + 'b', add_special_tokens=False)
            assert aa[:-1] == ids == bb[:-1]
            logits = model(mx.array(ids)[None, :])[:, -1, :].astype(mx.float32)[0]
            vals = logits[mx.array([aa[-1], bb[-1]])].tolist()
            f.write(json.dumps({'id': r['id'], 'prompt_sha256': r['prompt_sha256'],
                                'logit_a': vals[0], 'logit_b': vals[1]}) + '\n')
            if (i + 1) % 100 == 0:
                print(json.dumps({'rescored': i + 1, 'seconds': time.perf_counter() - start}), flush=True)
    print(json.dumps({'complete': True, 'seconds': time.perf_counter() - start}), flush=True)


if __name__ == '__main__':
    main()
