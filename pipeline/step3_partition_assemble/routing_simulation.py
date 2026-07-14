"""
Monte Carlo simulation of bucket-to-server routing strategies — C. briggsae
Reads timing data from Strategy B TSV and complexity scores from complexity TSV.
"""

import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import csv
from collections import defaultdict

# ==========================================
# Config — C. briggsae paths
# ==========================================
TIMING_TSV     = "/data3/projects/2025_Assembly/eyh/c_briggsae/assemblies/final_benchmark/strategyB_complexity_aware_40cpu/timing.tsv"
COMPLEXITY_TSV = "/data3/projects/2025_Assembly/eyh/c_briggsae/assemblies/bucket_complexity/bucket_complexity.tsv"
BASELINE_MIN   = 102   # C. briggsae baseline Trinity: 1h 42min
N_TRIALS       = 1000
SERVER_COUNTS  = [3, 4, 5, 6, 7, 8, 9, 10]
OUT_DIR        = "./routing_simulation_output_briggsae"
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================
# Load timing data from TSV
# ==========================================
def load_buckets(timing_tsv, complexity_tsv):
    # Load elapsed times
    times = {}
    with open(timing_tsv) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row['status'] == 'SUCCESS':
                bucket_id = int(row['bucket'])
                times[bucket_id] = float(row['elapsed_min'])

    # Load cpu_scores
    scores = {}
    with open(complexity_tsv) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            bucket_id = int(row['bucket'])
            try:
                scores[bucket_id] = float(row['cpu_score'])
            except (ValueError, KeyError):
                scores[bucket_id] = 0.0

    # Merge: (bucket_id, elapsed_min, cpu_score)
    buckets = []
    for bid, elapsed in times.items():
        cpu_score = scores.get(bid, 0.0)
        buckets.append((bid, elapsed, cpu_score))

    buckets.sort(key=lambda x: -x[1])  # sort by time descending
    return buckets

BUCKETS    = load_buckets(TIMING_TSV, COMPLEXITY_TSV)
SLOWEST    = max(b[1] for b in BUCKETS)
TOTAL_WORK = sum(b[1] for b in BUCKETS)

print("=" * 65)
print("  Monte Carlo Routing Simulation — C. briggsae")
print(f"  Buckets loaded : {len(BUCKETS)}")
print(f"  Total work     : {TOTAL_WORK:.1f} min  ({TOTAL_WORK/60:.2f}h)")
print(f"  Baseline       : {BASELINE_MIN:.0f} min  ({BASELINE_MIN/60:.2f}h)")
print(f"  Slowest bucket : {SLOWEST:.2f} min  ({SLOWEST/60:.2f}h)")
print(f"  Trials         : {N_TRIALS}")
print(f"  Server counts  : {SERVER_COUNTS}")
print(f"  Output dir     : {OUT_DIR}/")
print("=" * 65)

# Show loaded bucket data
print("\nLoaded bucket timings (top 10):")
print(f"  {'Bucket':>8} {'Time(min)':>10} {'CPU_Score':>10}")
print(f"  {'-'*8} {'-'*10} {'-'*10}")
for b in BUCKETS[:10]:
    print(f"  {b[0]:>8} {b[1]:>10.2f} {b[2]:>10.2f}")
print(f"  ... ({len(BUCKETS)} total)")

# ==========================================
# Scheduling strategies
# ==========================================
def wall_time(assignment):
    loads = defaultdict(float)
    for _, time, server in assignment:
        loads[server] += time
    return max(loads.values()) if loads else 0.0

def strategy_random(buckets, n_servers, seed=None):
    rng = random.Random(seed)
    return [(b[0], b[1], rng.randint(0, n_servers-1)) for b in buckets]

def strategy_greedy(buckets, n_servers):
    sorted_b = sorted(buckets, key=lambda x: -x[1])
    loads = [0.0] * n_servers
    assignment = []
    for b in sorted_b:
        s = loads.index(min(loads))
        assignment.append((b[0], b[1], s))
        loads[s] += b[1]
    return assignment

# def strategy_cpu_score(buckets, n_servers, seed=None):
#     rng = random.Random(seed)
#     loads = [0.0] * n_servers
#     assignment = []

#     # Determine tier thresholds from actual data
#     scores = [b[2] for b in buckets]
#     p75 = np.percentile(scores, 75)
#     p50 = np.percentile(scores, 50)

#     tier_high   = sorted([b for b in buckets if b[2] >= p75], key=lambda x: -x[1])
#     tier_medium = sorted([b for b in buckets if p50 <= b[2] < p75], key=lambda x: -x[1])
#     tier_low    = sorted([b for b in buckets if b[2] < p50], key=lambda x: -x[1])

#     hi_servers = list(range(min(2, n_servers)))
#     for b in tier_high:
#         s = min(hi_servers, key=lambda x: loads[x])
#         assignment.append((b[0], b[1], s))
#         loads[s] += b[1]
#     for b in tier_medium:
#         s = loads.index(min(loads))
#         assignment.append((b[0], b[1], s))
#         loads[s] += b[1]
#     for b in tier_low:
#         s = rng.randint(0, n_servers-1)
#         assignment.append((b[0], b[1], s))
#         loads[s] += b[1]
#     return assignment

def strategy_cpu_score(buckets, n_servers, seed=None):
    rng   = random.Random(seed)
    loads = [0.0] * n_servers
    assignment = []

    # Split by median complexity
    scores  = [b[2] for b in buckets]
    p50     = np.percentile(scores, 50)

    # Top 50% — greedy by cpu_score (deterministic, reliable)
    top_half = sorted([b for b in buckets if b[2] >= p50], key=lambda x: -x[2])
    for b in top_half:
        s = loads.index(min(loads))
        assignment.append((b[0], b[1], s))
        loads[s] += b[1]

    # Bottom 50% — random (stochastic, acknowledges uncertainty)
    bot_half = [b for b in buckets if b[2] < p50]
    for b in bot_half:
        s = rng.randint(0, n_servers - 1)
        assignment.append((b[0], b[1], s))
        loads[s] += b[1]

    return assignment


# ==========================================
# Run simulation for one server count
# ==========================================
def run_simulation(n_servers):
    results = {'random': [], 'greedy': [], 'cpu_score': []}
    results['greedy'].append(wall_time(strategy_greedy(BUCKETS, n_servers)))
    for trial in range(N_TRIALS):
        seed = trial * 42
        results['random'].append(wall_time(strategy_random(BUCKETS, n_servers, seed)))
        results['cpu_score'].append(wall_time(strategy_cpu_score(BUCKETS, n_servers, seed)))
    return results

# ==========================================
# Plot histogram
# ==========================================
def plot_histogram(times, strategy_name, n_servers, theoretical_min, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 4.5))

    arr = np.array(times) / 60
    baseline_h = BASELINE_MIN / 60
    theo_h     = theoretical_min / 60
    slowest_h  = SLOWEST / 60

    color_map = {'random': '#888780', 'greedy': '#1D9E75', 'cpu_score': '#378ADD'}
    c = color_map.get(strategy_name, '#888780')

    ax.hist(arr, bins=30, color=c, alpha=0.75, edgecolor='white', linewidth=0.4)
    ax.axvline(baseline_h,   color='#D85A30', lw=1.5, ls='--', label=f'Baseline Trinity ({baseline_h:.2f}h)')
    ax.axvline(theo_h,       color='#1D9E75', lw=1.5, ls=':',  label=f'Theoretical min ({theo_h:.2f}h)')
    ax.axvline(slowest_h,    color='#993C1D', lw=1.0, ls=':',  label=f'Slowest bucket floor ({slowest_h:.2f}h)')
    ax.axvline(np.mean(arr), color='#185FA5', lw=1.5, ls='-',  label=f'Mean ({np.mean(arr):.2f}h)')

    pct_beat = (arr < baseline_h).mean() * 100
    ax.set_title(
        f'C. briggsae — {strategy_name.replace("_"," ").title()} — {n_servers} servers\n'
        f'Mean={np.mean(arr):.2f}h  Std={np.std(arr):.2f}h  Beat baseline={pct_beat:.1f}%',
        fontsize=11
    )
    ax.set_xlabel('Wall time (hours)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    if standalone:
        fname = os.path.join(OUT_DIR, f'hist_{n_servers}servers_{strategy_name}.png')
        fig.tight_layout()
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return fname

def plot_combined(all_results, n_servers):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f'C. briggsae — Wall-time distribution — {n_servers} CPU servers\n'
        f'(N={N_TRIALS} trials, {len(BUCKETS)} buckets, {TOTAL_WORK:.0f} min total work)',
        fontsize=13, y=1.02
    )
    theo = TOTAL_WORK / n_servers
    for ax, strat in zip(axes, ['random', 'cpu_score', 'greedy']):
        plot_histogram(all_results[strat], strat, n_servers, theo, ax=ax)
    fname = os.path.join(OUT_DIR, f'combined_{n_servers}servers.png')
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fname

def plot_scaling_summary(summary):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    strategies = ['random', 'cpu_score', 'greedy']
    colors     = {'random': '#888780', 'cpu_score': '#378ADD', 'greedy': '#1D9E75'}
    labels     = {'random': 'Random', 'cpu_score': 'CPU-score aware', 'greedy': 'Greedy optimal'}
    x          = SERVER_COUNTS

    ax = axes[0]
    for s in strategies:
        means = [summary[n][s]['mean'] / 60 for n in x]
        ax.plot(x, means, 'o-', color=colors[s], label=labels[s], lw=2, ms=7)
    ax.axhline(BASELINE_MIN/60, color='#D85A30', ls='--', lw=1.5, label='Baseline Trinity')
    ax.axhline(SLOWEST/60,      color='#993C1D', ls=':',  lw=1.0, label='Floor (slowest bucket)')
    for n in x:
        ax.plot(n, TOTAL_WORK/n/60, 'x', color='#1D9E75', ms=10, mew=2)
    ax.set_xlabel('Number of CPU servers', fontsize=11)
    ax.set_ylabel('Mean wall time (hours)', fontsize=11)
    ax.set_title('C. briggsae — Mean wall time vs server count', fontsize=12)
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for s in strategies:
        speedups = [(BASELINE_MIN/60) / (summary[n][s]['mean']/60) for n in x]
        ax.plot(x, speedups, 'o-', color=colors[s], label=labels[s], lw=2, ms=7)
    ax.axhline(1.0, color='#D85A30', ls='--', lw=1.5, label='Baseline (1x)')
    theo_speedup = [(BASELINE_MIN/60) / (TOTAL_WORK/n/60) for n in x]
    ax.plot(x, theo_speedup, 'x--', color='#1D9E75', ms=10, mew=2, label='Theoretical max')
    ax.set_xlabel('Number of CPU servers', fontsize=11)
    ax.set_ylabel('Speedup vs baseline Trinity', fontsize=11)
    ax.set_title('C. briggsae — Speedup scaling with server count', fontsize=12)
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fname = os.path.join(OUT_DIR, 'scaling_summary.png')
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fname

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    summary = {}

    for n in SERVER_COUNTS:
        print(f"\n--- {n} servers ---")
        results = run_simulation(n)
        summary[n] = {}
        theo_min = TOTAL_WORK / n

        print(f"  {'Strategy':<16} {'Mean(h)':>8} {'Std(min)':>9} "
              f"{'Best(h)':>8} {'Worst(h)':>9} {'Beat baseline':>14} {'Speedup':>8}")
        print(f"  {'-'*16} {'-'*8} {'-'*9} {'-'*8} {'-'*9} {'-'*14} {'-'*8}")

        for strat, times in results.items():
            arr = np.array(times)
            pct = (arr < BASELINE_MIN).mean() * 100
            spd = BASELINE_MIN / arr.mean()
            summary[n][strat] = {
                'mean': arr.mean(), 'std': arr.std(),
                'min': arr.min(), 'max': arr.max(),
                'pct_beat': pct, 'speedup': spd,
            }
            print(f"  {strat:<16} {arr.mean()/60:>8.2f} {arr.std():>9.1f} "
                  f"{arr.min()/60:>8.2f} {arr.max()/60:>9.2f} "
                  f"{pct:>13.1f}% {spd:>8.2f}x")

        print(f"  Theoretical min: {theo_min/60:.2f}h  |  Floor: {SLOWEST/60:.2f}h")

        for strat, times in results.items():
            plot_histogram(times, strat, n, theo_min)
        plot_combined(results, n)

    plot_scaling_summary(summary)

    # Scaling ceiling analysis
    print("\n" + "=" * 65)
    print("  SCALING CEILING ANALYSIS — C. briggsae")
    print(f"  Slowest bucket : {SLOWEST:.2f} min = {SLOWEST/60:.2f}h (hard floor)")
    print(f"  Total work     : {TOTAL_WORK:.1f} min")
    print(f"  Baseline       : {BASELINE_MIN:.0f} min ({BASELINE_MIN/60:.2f}h)")
    print()
    print(f"  {'N servers':>10} {'Theo min(h)':>12} {'Greedy(h)':>10} "
          f"{'CPU-score(h)':>13} {'vs baseline':>12}")
    print(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*13} {'-'*12}")
    for n in SERVER_COUNTS:
        theo = TOTAL_WORK / n / 60
        g    = summary[n]['greedy']['mean'] / 60
        c    = summary[n]['cpu_score']['mean'] / 60
        spd  = BASELINE_MIN / 60 / c
        print(f"  {n:>10} {theo:>12.2f} {g:>10.2f} {c:>13.2f} {spd:>11.2f}x")

    print()
    print("  Extrapolated (greedy optimal):")
    print(f"  {'N servers':>10} {'Theo min(h)':>12} {'vs baseline':>12} {'Note'}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*25}")
    for n in [3, 4, 5, 6, 8, 10, 20, 50]:
        theo = TOTAL_WORK / n / 60
        spd  = BASELINE_MIN / 60 / max(theo, SLOWEST/60)
        note = '← floor reached' if theo < SLOWEST/60 else ''
        print(f"  {n:>10} {max(theo, SLOWEST/60):>12.2f} {spd:>11.2f}x  {note}")

    print(f"\n  Hard floor : {SLOWEST/60:.2f}h")
    print(f"  Max theoretical speedup: {BASELINE_MIN/SLOWEST:.2f}x")
    print("=" * 65)
    print(f"\nAll plots saved to: {OUT_DIR}/")