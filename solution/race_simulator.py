#!/usr/bin/env python
import json, sys
import numpy as np

# Paste from optimize_power_model.py output:
OFFSET = {'SOFT': -1.0891783687, 'MEDIUM': 0.0, 'HARD': 0.8656740398}    
BASE_DEG = {'SOFT': 0.5104952804, 'MEDIUM': 0.2461518193, 'HARD': 0.1270334226}
CLIFF = {'SOFT': 8.946984, 'MEDIUM': 18.688661, 'HARD': 28.847690}       
TEMP_COEFF = 0.0292926918
DEG_EXP = 1.2180192672

COMPOUND_TO_IDX = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
IDX_TO_COMPOUND = ["SOFT", "MEDIUM", "HARD"]

def build_stints(strategy, total_laps):
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

def simulate_race(test_case):
    cfg = test_case["race_config"]
    total_laps = int(cfg["total_laps"])
    base = float(cfg["base_lap_time"])
    pit = float(cfg["pit_lane_time"])
    temp = float(cfg["track_temp"])

    offsets = np.array([OFFSET["SOFT"], 0.0, OFFSET["HARD"]], dtype=np.float64)
    base_deg = np.array([BASE_DEG["SOFT"], BASE_DEG["MEDIUM"], BASE_DEG["HARD"]], dtype=np.float64)
    cliffs = np.array([CLIFF["SOFT"], CLIFF["MEDIUM"], CLIFF["HARD"]], dtype=np.float64)
    eff_deg = base_deg * (1.0 + TEMP_COEFF * temp)

    results = []

    for grid in range(1, 21):
        strat = test_case["strategies"][f"pos{grid}"]
        did = strat["driver_id"]

        stints, nstops = build_stints(strat, total_laps)

        # compute max stint length to build prefix sums
        maxL = max(L for _, L in stints if L > 0)
        ages = np.arange(1, maxL + 1, dtype=np.float64)

        prefix = np.zeros((3, maxL), dtype=np.float64)
        for comp in range(3):
            f = np.maximum(0.0, ages - cliffs[comp]) ** DEG_EXP
            prefix[comp] = np.cumsum(f)

        tt = nstops * pit
        for comp, L in stints:
            if L <= 0:
                continue
            tt += L * (base + offsets[comp])
            tt += eff_deg[comp] * prefix[comp, L - 1]

        results.append((tt, grid, did))

    results.sort()
    return [r[2] for r in results]

def main():
    try:
        data = sys.stdin.read()
        if not data.strip():
            return
        test_case = json.loads(data)
        out = {
            "race_id": test_case.get("race_id", ""),
            "finishing_positions": simulate_race(test_case)
        }
        sys.stdout.write(json.dumps(out) + "\n")
    except Exception:
        pass

if __name__ == "__main__":
    main()