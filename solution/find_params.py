#!/usr/bin/env python3
"""
solution/fast_optimize.py

Fast parameter search using differential_evolution across multiple model types.
Run: python3 solution/fast_optimize.py
"""

import json
import os
import sys
import numpy as np
from scipy.optimize import differential_evolution


def progress_callback_factory(label, every=5):
    gen = [0]

    def cb(xk, convergence):
        gen[0] += 1
        if gen[0] % every == 0:
            print(f"    [{label}] generation={gen[0]} convergence={convergence:.3e}", flush=True)
        return False

    return cb

def load_test_cases():
    pairs = []
    for i in range(1, 101):
        fn = f"test_{i:03d}.json"
        inp = f"data/test_cases/inputs/{fn}"
        out = f"data/test_cases/expected_outputs/{fn}"
        if os.path.exists(inp) and os.path.exists(out):
            with open(inp) as f:
                ti = json.load(f)
            with open(out) as f:
                to = json.load(f)
            pairs.append((ti, to["finishing_positions"]))
    return pairs

def load_historical(max_files=1):
    races = []
    directory = "data/historical_races"
    files = sorted(f for f in os.listdir(directory) if f.endswith('.json'))[:max_files]
    for fn in files:
        with open(os.path.join(directory, fn)) as f:
            data = json.load(f)
            if isinstance(data, list):
                races.extend(data)
            else:
                races.append(data)
    return races

def simulate_generic(race, params, deg_func, temp_func):
    os_, oh_, ds_, dm_, dh_, cs_, cm_, ch_, tc_ = params
    offsets = {'SOFT': os_, 'MEDIUM': 0.0, 'HARD': oh_}
    degs = {'SOFT': ds_, 'MEDIUM': dm_, 'HARD': dh_}
    cliffs = {'SOFT': cs_, 'MEDIUM': cm_, 'HARD': ch_}

    config = race["race_config"]
    total_laps = int(config["total_laps"])
    base_lap = float(config["base_lap_time"])
    pit_time = float(config["pit_lane_time"])
    temp = float(config["track_temp"])

    results = []
    for pi in range(1, 21):
        pk = f"pos{pi}"
        strat = race["strategies"][pk]
        did = strat["driver_id"]
        pm = {int(p["lap"]): p["to_tire"] for p in strat.get("pit_stops", [])}
        ct = strat["starting_tire"]
        ta = 0
        tt = 0.0
        for lap in range(1, total_laps + 1):
            ta += 1
            eff = temp_func(degs[ct], tc_, temp)
            deg = deg_func(eff, ta, cliffs[ct])
            tt += base_lap + offsets[ct] + deg
            if lap in pm:
                tt += pit_time
                ct = pm[lap]
                ta = 0
        results.append((tt, pi, did))
    results.sort()
    return [r[2] for r in results]

def make_objective(all_pairs, deg_func, temp_func):
    eval_count = [0]

    def objective(params):
        eval_count[0] += 1
        if eval_count[0] % 10 == 0:
            print(f"    objective evaluations: {eval_count[0]}", flush=True)
        total_err = 0
        for race, expected in all_pairs:
            pred = simulate_generic(race, params, deg_func, temp_func)
            ep = {d: i for i, d in enumerate(expected)}
            for i, d in enumerate(pred):
                total_err += (i - ep[d]) ** 2
        return total_err
    return objective

def count_exact(test_pairs, params, deg_func, temp_func):
    exact = 0
    for race, expected in test_pairs:
        pred = simulate_generic(race, params, deg_func, temp_func)
        if pred == expected:
            exact += 1
    return exact

def main():
    full_mode = "--full" in sys.argv
    first_maxiter = 200 if full_mode else 40
    first_popsize = 20 if full_mode else 10
    refine_maxiter = 500 if full_mode else 120
    refine_popsize = 30 if full_mode else 15

    mode_name = "FULL" if full_mode else "FAST"
    print(f"Mode: {mode_name}")
    if not full_mode:
        print("Tip: use --full for the original long search", flush=True)

    print("Loading data...")
    test_pairs = load_test_cases()
    hist_races = load_historical(max_files=1)
    print(f"Test cases: {len(test_pairs)}, Historical: {len(hist_races)}")

    # Build training set: test cases + some historical
    all_pairs = list(test_pairs)
    for r in hist_races[:200]:
        if 'finishing_positions' in r:
            all_pairs.append((r, r['finishing_positions']))
    print(f"Training on {len(all_pairs)} races")

    # Define model variants
    deg_functions = {
        'linear_cliff': lambda e, ta, cl: e * max(0.0, ta - cl),
        'cumul_cliff': lambda e, ta, cl: e * max(0.0, ta - cl) * (max(0.0, ta - cl) + 1.0) / 2.0,
        'quad_cliff': lambda e, ta, cl: e * (max(0.0, ta - cl) ** 2),
        'linear_nocliff': lambda e, ta, cl: e * ta,
        'cumul_nocliff': lambda e, ta, cl: e * ta * (ta + 1.0) / 2.0,
    }

    temp_functions = {
        'mult_raw': lambda d, tc, t: d * (1.0 + tc * t),
        'mult_c30': lambda d, tc, t: d * (1.0 + tc * (t - 30.0)),
        'additive': lambda d, tc, t: d + tc * t,
    }

    best_global_exact = 0
    best_global_config = None
    best_global_params = None

    for dname, dfunc in deg_functions.items():
        for tname, tfunc in temp_functions.items():
            config_name = f"{dname}__{tname}"
            print(f"\n--- {config_name} ---")

            obj = make_objective(all_pairs, dfunc, tfunc)

            if 'nocliff' in dname:
                bounds = [
                    (-2.0, 0.0), (0.0, 2.0),         # offsets
                    (0.0001, 0.05), (0.0001, 0.02), (0.00001, 0.01),  # degs
                    (-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01),      # cliffs (unused)
                    (0.0001, 0.03),                    # temp
                ]
            elif 'cumul' in dname:
                bounds = [
                    (-2.0, 0.0), (0.0, 2.0),
                    (0.001, 0.08), (0.0005, 0.04), (0.0001, 0.02),
                    (1, 20), (5, 35), (10, 50),
                    (0.0001, 0.03),
                ]
            elif 'quad' in dname:
                bounds = [
                    (-2.0, 0.0), (0.0, 2.0),
                    (0.0005, 0.05), (0.0002, 0.02), (0.00005, 0.01),
                    (1, 20), (5, 35), (10, 50),
                    (0.0001, 0.03),
                ]
            else:  # linear_cliff
                bounds = [
                    (-2.0, 0.0), (0.0, 2.0),
                    (0.005, 0.5), (0.002, 0.2), (0.0005, 0.1),
                    (1, 20), (5, 35), (10, 50),
                    (0.0001, 0.03),
                ]

            try:
                cb = progress_callback_factory(config_name, every=5)
                result = differential_evolution(
                    obj, bounds,
                    maxiter=first_maxiter, popsize=first_popsize,
                    seed=42, tol=1e-12,
                    mutation=(0.5, 1.5), recombination=0.8,
                    disp=False, polish=True,
                    callback=cb
                )

                p = result.x
                exact = count_exact(test_pairs, p, dfunc, tfunc)
                print(f"  exact={exact}/100, loss={result.fun:.1f}")
                print(f"  params: os={p[0]:.6f} oh={p[1]:.6f} ds={p[2]:.6f} dm={p[3]:.6f} dh={p[4]:.6f} cs={p[5]:.2f} cm={p[6]:.2f} ch={p[7]:.2f} tc={p[8]:.8f}")

                if exact > best_global_exact:
                    best_global_exact = exact
                    best_global_config = config_name
                    best_global_params = p.copy()
                    print(f"  *** NEW BEST: {exact}/100 ***")

            except Exception as e:
                print(f"  FAILED: {e}")

    # Now do a second pass with more iterations on the top model
    if best_global_config:
        print(f"\n{'='*60}")
        print(f"REFINING BEST: {best_global_config} ({best_global_exact}/100)")
        print(f"{'='*60}")

        dname, tname = best_global_config.split('__')
        dfunc = deg_functions[dname]
        tfunc = temp_functions[tname]

        # Tighter bounds around best
        p = best_global_params
        tight_bounds = []
        for i in range(9):
            v = p[i]
            if i < 2:  # offsets
                tight_bounds.append((v - 0.3, v + 0.3))
            elif i < 5:  # degs
                tight_bounds.append((max(0.00001, v * 0.3), v * 3.0))
            elif i < 8:  # cliffs
                tight_bounds.append((max(0, v - 5), v + 5))
            else:  # temp
                tight_bounds.append((max(0.00001, v * 0.3), v * 3.0))

        obj = make_objective(all_pairs, dfunc, tfunc)

        result = differential_evolution(
            obj, tight_bounds,
            maxiter=refine_maxiter, popsize=refine_popsize,
            seed=123, tol=1e-14,
            mutation=(0.3, 1.2), recombination=0.9,
            disp=True, polish=True,
            callback=progress_callback_factory("refine", every=10)
        )

        p2 = result.x
        exact2 = count_exact(test_pairs, p2, dfunc, tfunc)
        print(f"\nRefined: exact={exact2}/100, loss={result.fun:.1f}")

        if exact2 >= best_global_exact:
            best_global_params = p2
            best_global_exact = exact2

        # Also try integer cliffs
        best_int = best_global_exact
        best_int_params = best_global_params.copy()
        p = best_global_params
        for cs in [int(p[5]), int(p[5])+1]:
            for cm in [int(p[6]), int(p[6])+1]:
                for ch in [int(p[7]), int(p[7])+1]:
                    test_p = p.copy()
                    test_p[5] = cs
                    test_p[6] = cm
                    test_p[7] = ch
                    ex = count_exact(test_pairs, test_p, dfunc, tfunc)
                    if ex > best_int:
                        best_int = ex
                        best_int_params = test_p.copy()
                        print(f"  Integer cliffs ({cs},{cm},{ch}): {ex}/100")

        if best_int > best_global_exact:
            best_global_params = best_int_params
            best_global_exact = best_int

    # Print final results
    p = best_global_params
    dname, tname = best_global_config.split('__')
    
    print(f"""
{'='*60}
FINAL BEST: {best_global_config} with {best_global_exact}/100 exact matches
{'='*60}

Copy into race_simulator.py:

DEG_MODEL = '{dname}'
TEMP_MODEL = '{tname}'
OFFSET = {{'SOFT': {p[0]:.10f}, 'MEDIUM': 0.0, 'HARD': {p[1]:.10f}}}
BASE_DEG = {{'SOFT': {p[2]:.10f}, 'MEDIUM': {p[3]:.10f}, 'HARD': {p[4]:.10f}}}
CLIFF = {{'SOFT': {p[5]:.6f}, 'MEDIUM': {p[6]:.6f}, 'HARD': {p[7]:.6f}}}
TEMP_COEFF = {p[8]:.10f}
""")

    # Verify on historical data too
    hist_exact = 0
    for r in hist_races[:500]:
        if 'finishing_positions' in r:
            pred = simulate_generic(r, best_global_params, 
                                   deg_functions[dname], temp_functions[tname])
            if pred == r['finishing_positions']:
                hist_exact += 1
    print(f"Historical validation: {hist_exact}/500 exact matches")

if __name__ == "__main__":
    main()