"""Independent integrity, budget, and Monte Carlo checks for the saved pilot."""
import csv
import hashlib
import json
from pathlib import Path
from decimal import Decimal, localcontext

import numpy as np
from analyze_pilot import allocation, attack, residual

ROOT = Path(__file__).resolve().parent


def main():
    metadata = json.loads((ROOT / 'run_metadata.json').read_text())
    assert hashlib.sha256((ROOT / 'protocol.json').read_bytes()).hexdigest() == metadata['protocol_sha256']
    for path, digest in metadata['input_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    prompts = {r['id']: r for r in map(json.loads, (ROOT / 'prompts.jsonl').read_text().splitlines())}
    rows = list(map(json.loads, (ROOT / 'judge_outputs.jsonl').read_text().splitlines()))
    assert len(rows) == len(prompts) == 600
    assert len({r['id'] for r in rows}) == 600
    for r in rows:
        p = prompts[r['id']]
        assert r['prompt_sha256'] == hashlib.sha256(p['prompt'].encode()).hexdigest()
        assert r['prompt_sha256'] == p['prompt_sha256']
        assert np.isfinite([r['p_a'], r['p_b'], r['label_mass'], r['total_seconds']]).all()
        assert abs(r['p_a'] + r['p_b'] - 1) < 0.000001
        assert 0 <= r['label_mass'] <= 1.00001
        assert r['target_score'] == r['p_' + r['target_label']]
        assert r['source_substring_success'] == (r['target_string'] in r['raw_output'])
    replay = list(csv.DictReader((ROOT / 'replay_summary.csv').open()))
    for r in replay:
        subset = [x for x in rows if x['condition'] == r['condition']]
        assert len(subset) == 200
        assert abs(float(r['free_target_selection_rate']) - np.mean([x['source_substring_success'] for x in subset])) < 1e-12
    logits = {r['id']: r for r in map(json.loads, (ROOT / 'choice_logits.jsonl').read_text().splitlines())}
    assert len(logits) == 600
    lookup = {(r['question_id'], r['competitor_index'], r['orientation'], r['condition']): r for r in rows}
    with localcontext() as ctx:
        ctx.prec = 100
        for p in csv.DictReader((ROOT / 'pair_scores.csv').open()):
            margins = []
            for orientation in [0, 50]:
                r = lookup[(int(p['question_id']), int(p['competitor_index']), orientation, p['condition'])]
                ll = logits[r['id']]
                margin = Decimal(str(ll['logit_a'])) - Decimal(str(ll['logit_b']))
                if r['target_label'] == 'b':
                    margin = -margin
                margins.append(margin)
            average = sum(Decimal(1) / (Decimal(1) + (-m).exp()) for m in margins) / 2
            if sum(margins) == 0:
                assert abs(average - Decimal('0.5')) < Decimal('1e-90')
                expected = False
            else:
                expected = average > Decimal('0.5')
            assert bool(int(p['target_chosen'])) == expected, p
    q = np.array([0.01, 0.2, 0.49, 0.8, 0.99])
    rng = np.random.default_rng(47)
    audit_checks = []
    for method in ['entropy', 'uniform', 'mixture_50']:
        analytic, p = residual(q, method, 0.2)
        assert abs(p.mean() - 0.2) < 1e-10
        assert np.allclose(allocation(np.tile(q, (3, 1)), method, 0.2)[0], p)
        coins = rng.random((200000, len(q))) < p
        simulation = np.mean((q > 0.5) * ~coins)
        assert abs(simulation - analytic.mean()) < 0.003
        mask, _, _ = attack(q, 1 - q, method, 0.2, 2)
        assert mask.sum() <= 2
        assert residual(np.where(mask, 1 - q, q), method, 0.2)[0].mean() >= analytic.mean() - 1e-10
        audit_checks.append({'allocator': method, 'analytic': float(analytic.mean()), 'simulation': float(simulation)})
    for q in [np.zeros(100), np.ones(100), np.full(100, 0.5)]:
        assert np.allclose(allocation(q, 'entropy', 0.2), 0.2)
    for r in csv.DictReader((ROOT / 'adaptive_audits.csv').open()):
        assert int(r['actually_attacked_pairs']) <= round(100 * float(r['corruption_fraction']))
        assert abs(float(r['expected_audits']) - 100 * float(r['audit_fraction'])) < 1e-8
    report = {'status': 'passed', 'records_verified': 600, 'question_clusters': 10,
              'pair_decisions_checked_with_100_digit_arithmetic': 300,
              'monte_carlo_cross_checks': audit_checks,
              'meaning': 'Integrity and numerical checks; not independent reproduction of model outputs or a statistical guarantee.'}
    (ROOT / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
