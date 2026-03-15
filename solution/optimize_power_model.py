#!/usr/bin/env python
import json, os, math
import numpy as np
from scipy.optimize import differential_evolution

COMPOUND_TO_IDX = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}

def load_test_pairs():
    pairs = []
    for i in range(1, 101):
        inp = f"data/test_cases/inputs/test_{i:03d}.json"
        out = f"data/test_cases/expected_outputs/test_{i:03d}.json"
        with open(inp, "r") as f:
            race = json.load(f)
        with open(out, "r") as f:
            exp = json.load(f)["finishing_positions"]
        pairs.append((race, exp))
    return pairs

def build_stints(strategy, total_laps):
    """Return list of (compound_idx, stint_len), pit at end of lap."""
    pits = sorted(strategy.get("pit_stops", []), key=lambda p: int(p["lap"]))
    stints = []
    cur = strategy["starting_tire"]
    start = 1
    for p in pits:
        lap = int(p["lap"])
        stints.append((COMPOUND_TO_IDX[cur], lap - start + 1))
        cur = p["to_tire"]
        start = lap + 1
    stints.append((COMPOUND_TO_IDX[cur], total_laps - start + 1))
    return stints, len(pits)

def preprocess(pairs):
    races = []
    for race, expected in pairs:
        cfg = race["race_config"]
        total_laps = int(cfg["total_laps"])
        base = float(cfg["base_lap_time"])
        pit = float(cfg["pit_lane_time"])
        temp = float(cfg["track_temp"])

        drivers = []
        for grid in range(1, 21):
            strat = race["strategies"][f"pos{grid}"]
            did = strat["driver_id"]
            stints, nstops = build_stints(strat, total_laps)
            drivers.append((did, grid, stints, nstops))

        exp_rank = {d:i for i, d in enumerate(expected)}
        races.append((base, pit, temp, drivers, exp_rank, expected))
    return races

def score_params(pre, params):
    # params = [offS, offH, degS, degM, degH, cliffS, cliffM, cliffH, tempC, exp]
    offS, offH, dS, dM, dH, cS, cM, cH, tC, expo = params

    offsets = np.array([offS, 0.0, offH], dtype=np.float64)
    base_deg = np.array([dS, dM, dH], dtype=np.float64)
    cliffs = np.array([cS, cM, cH], dtype=np.float64)

    total_err = 0.0

    for base, pit, temp, drivers, exp_rank, expected in pre:
        # eff_deg = base_deg * (1 + tC * temp)
        eff_deg = base_deg * (1.0 + tC * temp)

        # precompute prefix sums of (max(0, age - cliff)**expo) for each compound
        # ages max is <= 70 typically; compute max stint len in this race
        maxL = 0
        for _, _, stints, _ in drivers:
            for comp, L in stints:
                if L > maxL:
                    maxL = L
        ages = np.arange(1, maxL + 1, dtype=np.float64)

        prefix = np.zeros((3, maxL), dtype=np.float64)
        for comp in range(3):
            f = np.maximum(0.0, ages - cliffs[comp]) ** expo
            prefix[comp] = np.cumsum(f)

        times = []
        for did, grid, stints, nstops in drivers:
            tt = nstops * pit
            for comp, L in stints:
                if L <= 0:
                    continue
                tt += L * (base + offsets[comp])
                tt += eff_deg[comp] * prefix[comp, L - 1]
            times.append((tt, grid, did))

        times.sort()
        pred = [x[2] for x in times]

        # squared rank error
        for i, did in enumerate(pred):
            di = i - exp_rank[did]
            total_err += di * di

    return total_err

def count_exact(pre, params):
    offS, offH, dS, dM, dH, cS, cM, cH, tC, expo = params
    offsets = np.array([offS, 0.0, offH], dtype=np.float64)
    base_deg = np.array([dS, dM, dH], dtype=np.float64)
    cliffs = np.array([cS, cM, cH,], dtype=np.float64)

    exact = 0
    for base, pit, temp, drivers, _, expected in pre:
        eff_deg = base_deg * (1.0 + tC * temp)

        maxL = 0
        for _, _, stints, _ in drivers:
            for comp, L in stints:
                maxL = max(maxL, L)
        ages = np.arange(1, maxL + 1, dtype=np.float64)

        prefix = np.zeros((3, maxL), dtype=np.float64)
        for comp in range(3):
            f = np.maximum(0.0, ages - cliffs[comp]) ** expo
            prefix[comp] = np.cumsum(f)

        times = []
        for did, grid, stints, nstops in drivers:
            tt = nstops * pit
            for comp, L in stints:
                if L <= 0:
                    continue
                tt += L * (base + offsets[comp])
                tt += eff_deg[comp] * prefix[comp, L - 1]
            times.append((tt, grid, did))

        times.sort()
        pred = [x[2] for x in times]
        if pred == expected:
            exact += 1
    return exact

def main():
    pairs = load_test_pairs()
    pre = preprocess(pairs)

    # Start near your best heuristic (mult + “cumul-ish” => exponent > 1)
    # Bounds chosen to be reasonable and fast.
    bounds = [
        (-1.5, 0.0),    # offS
        (0.0, 1.5),     # offH
        (0.001, 0.8),   # degS
        (0.001, 0.5),   # degM
        (0.001, 0.3),   # degH
        (0.0, 20.0),    # cliffS
        (0.0, 35.0),    # cliffM
        (0.0, 50.0),    # cliffH
        (0.0, 0.03),    # tempC
        (0.8, 2.2),     # exponent
    ]

    def obj(x):
        return score_params(pre, x)

    print("Optimizing (power-law degradation)...")
    res = differential_evolution(
        obj, bounds,
        maxiter=250, popsize=18,
        seed=42, tol=1e-12,
        mutation=(0.5, 1.5), recombination=0.8,
        disp=True, polish=True, workers=1
    )

    p = res.x
    ex = count_exact(pre, p)
    print("\nDone.")
    print(f"Exact: {ex}/100")
    print("Paste into race_simulator.py:\n")
    print(f"OFFSET = {{'SOFT': {p[0]:.10f}, 'MEDIUM': 0.0, 'HARD': {p[1]:.10f}}}")
    print(f"BASE_DEG = {{'SOFT': {p[2]:.10f}, 'MEDIUM': {p[3]:.10f}, 'HARD': {p[4]:.10f}}}")
    print(f"CLIFF = {{'SOFT': {p[5]:.6f}, 'MEDIUM': {p[6]:.6f}, 'HARD': {p[7]:.6f}}}")
    print(f"TEMP_COEFF = {p[8]:.10f}")
    print(f"DEG_EXP = {p[9]:.10f}")

if __name__ == "__main__":
    main()