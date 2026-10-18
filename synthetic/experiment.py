#!/usr/bin/env python3
"""Synthetic judge-score/audit-priority attacks. No LLMs or real-world claims.

The practical normal-approximation gate is deliberately evaluated, not certified.
A separate conservative Bernstein gate supplies a valid fixed-pool reference.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_name, '1')
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import NormalDist
import time
import numpy as np

DEFENDERS = {'uniform': 1.0, 'active': 0.0, 'mix25': 0.25, 'mix50': 0.50, 'mix75': 0.75}
GATES = ('point', 'normal', 'bernstein')

@dataclass
class Pool:
    labels: np.ndarray
    scores: np.ndarray
    priorities: np.ndarray
    group: np.ndarray


def rng_for(seed, stream):
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))


def make_pool(n, seed, stream=0):
    rng = rng_for(seed, stream)
    group = (rng.random(n) < 0.2).astype(int)
    # Shared outcome noise makes candidate-versus-baseline comparisons paired.
    accuracy = np.array([0.62, 0.68, 0.60])[None, :] - 0.18 * group[:, None]
    labels = (rng.random((n, 1)) < accuracy).astype(float)
    error_rate = np.where(group == 0, 0.01, 0.25)
    shared_flip = rng.random(n) < error_rate
    independent_flip = rng.random((n, 3)) < error_rate[:, None]
    use_independent = rng.random((n, 3)) < 0.2
    flips = np.where(use_independent, independent_flip, shared_flip[:, None])
    proxy = np.logical_xor(labels.astype(bool), flips)
    scores = 0.02 + 0.96 * proxy.astype(float)
    # An imperfect acquisition model: expected error by visible difficulty group.
    priorities = np.sqrt(error_rate + 0.002)
    return Pool(labels, scores, priorities, group)


def fit_surrogate(calibration):
    """Independent calibration labels only; no evaluation labels enter attack search."""
    table = np.empty((2, 2, 3))
    bits = calibration.scores > 0.5
    for g, bit, j in itertools.product(range(2), range(2), range(3)):
        mask = (calibration.group == g) & (bits[:, j] == bit)
        table[g, bit, j] = (calibration.labels[mask, j].sum() + 1) / (mask.sum() + 2)
    return table


def predict_labels(scores, group, table):
    return np.column_stack([table[group, (scores[:, j] > 0.5).astype(int), j] for j in range(3)])


def allocate(priority, fraction, mix):
    """Water-filled active probabilities, then a uniform mixture; exact expected budget."""
    if not 0 < fraction <= 1 or not 0 <= mix <= 1:
        raise ValueError('Invalid fraction/mix')
    n = len(priority)
    if fraction == 1 or mix == 1:
        return np.full(n, fraction)
    weights = np.maximum(np.asarray(priority, dtype=float), 1e-5)
    low, high = 0.0, 1.0 / weights.min()
    for _ in range(80):
        mid = (low + high) / 2
        if np.minimum(1, mid * weights).mean() < fraction:
            low = mid
        else:
            high = mid
    active = np.minimum(1, (low + high) / 2 * weights)
    return mix * fraction + (1 - mix) * active


def contrasts(values):
    return values[:, 1:] - values[:, [0]]


def estimates(labels, scores, p, coins):
    d, s = contrasts(labels), contrasts(scores)
    residual = d - s
    audits = coins < p
    n = len(p)
    est = s.mean(axis=0) + audits @ (residual / p[:, None]) / n
    # Unbiased design-variance estimate, but a normal CI is NOT guaranteed valid.
    var = audits @ ((1 - p[:, None]) * residual**2 / p[:, None]**2) / n**2
    return est, np.maximum(var, 0), audits


def bounds(est, var, p, gate, alpha=0.05):
    k = est.shape[1]
    if gate == 'point':
        return est
    if gate == 'normal':
        z = NormalDist().inv_cdf(1 - alpha / k)
        return est - z * np.sqrt(var)
    if gate != 'bernstein':
        raise ValueError(gate)
    if np.all(p == 1):
        return est
    # Fixed-pool residual differences lie in [-2,2]. Independent audit coins.
    n, t = len(p), math.log(k / alpha)
    v = 4 * np.sum(1 / p - 1)
    m = 2 / p.min()
    return est - (math.sqrt(2 * v * t) + 2 * m * t / 3) / n


def select(lower):
    selected = 1 + lower.argmax(axis=1)
    selected[lower.max(axis=1) <= 0] = 0
    return selected


def evaluate(labels, scores, p, coins, gate):
    est, var, audits = estimates(labels, scores, p, coins)
    low = bounds(est, var, p, gate)
    selected = select(low)
    delta = np.r_[0.0, contrasts(labels).mean(axis=0)]
    realized = delta[selected]
    return {
        'harm': float(np.mean(realized < -0.01)),
        'harm_any': float(np.mean(realized < 0)),
        'utility': float(realized.mean()),
        'fallback': float(np.mean(selected == 0)),
        'selected_bad': float(np.mean(selected == 2)),
        'lower_coverage': float(np.mean(np.all(low <= delta[1:] + 1e-12, axis=1))),
        'opportunity_loss': float(delta.max() - realized.mean()),
        'audit_fraction': float(audits.mean()),
        'gold_labels_mean': float(3 * audits.sum(axis=1).mean()),
        'estimator_bias_good': float(est[:, 0].mean() - delta[1]),
        'estimator_bias_bad': float(est[:, 1].mean() - delta[2]),
        'variance_bad_mc': float(np.var(est[:, 1])),
        'variance_bad_exact': float(np.sum((1 / p - 1) * (contrasts(labels)[:, 1] - contrasts(scores)[:, 1])**2) / len(p)**2),
        'true_good_gain': float(delta[1]), 'true_bad_gain': float(delta[2]),
        'min_audit_probability': float(p.min()),
        'max_ipw': float(1 / p.min()),
    }


def attack_candidates(scores, priority, group, fraction, seed, count):
    """Candidate family uses public proxies only, never evaluation truth."""
    rng = rng_for(seed, 20)
    n, target = len(scores), 2
    k = int(math.floor(fraction * n))
    candidates = []
    # Include fixed reference: raise lowest-scored candidate outputs to one.
    for c in range(count):
        strategy = c % 4
        noise = rng.random(n) * 0.05
        if strategy == 0:
            rank = scores[:, target] + noise
        elif strategy == 1:
            rank = scores[:, target] + 0.2 * group + noise
        elif strategy == 2:
            rank = scores[:, target] - 0.2 * group + noise
        else:
            rank = rng.random(n)
        indices = np.argsort(rank, kind='stable')[:k]
        level = (1.0, 0.9, 0.75)[(c // 4) % 3]
        multiplier = (0.001, 0.03, 0.3, 1.0, 10.0)[(c // 12) % 5]
        if c == 0:
            level, multiplier = 1.0, 0.001
        new_scores, new_priority = scores.copy(), priority.copy()
        new_scores[indices, target] = level
        new_priority[indices] *= multiplier
        candidates.append((new_scores, new_priority, {'candidate': c, 'indices': indices.tolist(), 'score': level, 'priority_multiplier': multiplier}))
    return candidates


def search_attack(candidates, surrogate, fraction, mix, coins, objective):
    """Adaptive to the defender; surrogate labels and search-only coins."""
    delta = np.r_[0.0, contrasts(surrogate).mean(axis=0)]
    best, best_value = None, -np.inf
    for scores, priority, record in candidates:
        p = allocate(priority, fraction, mix)
        est, var, _ = estimates(surrogate, scores, p, coins)
        chosen = select(bounds(est, var, p, 'normal'))
        if objective == 'harm':
            value = np.mean(chosen == 2)
        elif objective == 'utility':
            value = delta.max() - delta[chosen].mean()
        else:
            raise ValueError(objective)
        # Deterministic tie break; actual test labels and test coins never used.
        value += 1e-9 * contrasts(scores)[:, 1].mean()
        if value > best_value:
            best, best_value = (scores, priority, dict(record)), value
    best[2]['surrogate_objective'] = float(best_value)
    best[2]['objective'] = objective
    return best


def run_seed(args):
    cfg, seed = args
    start = time.perf_counter()
    pool = make_pool(cfg['n'], seed)
    calibration = make_pool(cfg['calibration_n'], seed, 1)
    surrogate = predict_labels(pool.scores, pool.group, fit_surrogate(calibration))
    search_coins = rng_for(seed, 30).random((cfg['search_replays'], cfg['n']))
    coins = rng_for(seed, 40).random((cfg['replays'], cfg['n']))
    candidates = attack_candidates(pool.scores, pool.priorities, pool.group, cfg['attack_fraction'], seed, cfg['search_candidates'])
    rows, witness = [], {}
    for name, mix in DEFENDERS.items():
        harm_attack = search_attack(candidates, surrogate, cfg['audit_fraction'], mix, search_coins, 'harm')
        utility_attack = search_attack(candidates, surrogate, cfg['audit_fraction'], mix, search_coins, 'utility')
        # Four cells use the SAME payload optimized against this defender.
        af, aq, meta = harm_attack
        conditions = {
            'clean': (pool.scores, pool.priorities),
            'random_joint': candidates[3][:2],
            'scores_only': (af, pool.priorities),
            'priorities_only': (pool.scores, aq),
            'joint_harm': (af, aq),
            'joint_utility': utility_attack[:2],
        }
        for condition, (f, q) in conditions.items():
            p = allocate(q, cfg['audit_fraction'], mix)
            for gate in GATES:
                metrics = evaluate(pool.labels, f, p, coins, gate)
                rows.append(dict(seed=seed, defender=name, condition=condition, gate=gate, **metrics))
        if name == 'active':
            indices = np.array(meta['indices'])
            pclean = allocate(pool.priorities, cfg['audit_fraction'], mix)
            patt = allocate(aq, cfg['audit_fraction'], mix)
            witness = {
                'seed': seed, 'attack': meta,
                'payload_changed_rows': int(np.sum(np.any(af != pool.scores, axis=1) | (aq != pool.priorities))),
                'permitted_rows': int(cfg['n'] * cfg['attack_fraction']),
                'clean_mean_p_on_edited': float(pclean[indices].mean()),
                'attacked_mean_p_on_edited': float(patt[indices].mean()),
                'examples': [dict(row=int(i), group=int(pool.group[i]), true_labels=pool.labels[i].tolist(), clean_scores=pool.scores[i].tolist(), attacked_scores=af[i].tolist(), clean_p=float(pclean[i]), attacked_p=float(patt[i])) for i in indices[:5]],
                'truth_access_note': 'True labels are added by evaluator after attack commitment, not supplied to search.'
            }
    # No-audit anchors, charged for zero audit labels.
    for cond, f in [('clean', pool.scores), ('joint_harm', candidates[0][0])]:
        chosen = int(1 + np.argmax(contrasts(f).mean(axis=0)))
        if contrasts(f).mean(axis=0).max() <= 0:
            chosen = 0
        delta = np.r_[0.0, contrasts(pool.labels).mean(axis=0)]
        rows.append(dict(seed=seed, defender='raw_judge', condition=cond, gate='raw', harm=float(delta[chosen]<-.01), harm_any=float(delta[chosen]<0), utility=float(delta[chosen]), fallback=float(chosen==0), selected_bad=float(chosen==2), lower_coverage=float('nan'), opportunity_loss=float(delta.max()-delta[chosen]), audit_fraction=0.0, gold_labels_mean=0.0, true_good_gain=float(delta[1]), true_bad_gain=float(delta[2])))
    return rows, witness, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config_pilot.json')
    parser.add_argument('--out', default='results')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--seed-start', type=int, default=None)
    parser.add_argument('--seeds', type=int, default=None)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.seed_start is not None: cfg['seed_start'] = args.seed_start
    if args.seeds is not None: cfg['seeds'] = args.seeds
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    jobs = [(cfg, seed) for seed in range(cfg['seed_start'], cfg['seed_start'] + cfg['seeds'])]
    if args.workers == 1:
        results = [run_seed(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(run_seed, jobs))
    rows = [row for result in results for row in result[0]]
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (out/'seed_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    (out/'witness.json').write_text(json.dumps(results[0][1], indent=2)+'\n')
    metadata = dict(config=cfg, workers=args.workers, elapsed_seconds=time.perf_counter()-started,
                    seed_cpu_seconds=[r[2] for r in results], numpy_version=np.__version__,
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                    real_llm_experiments=False, published_method_reproductions=False,
                    normal_gate_is_certified=False, attacker_calibration_labels_per_pool=3*cfg['calibration_n'])
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'output':str(out.resolve()),'rows':len(rows),'elapsed_seconds':metadata['elapsed_seconds']}))

if __name__ == '__main__':
    main()
