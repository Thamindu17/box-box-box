#!/usr/bin/env python
import json, sys
try:
    import numpy as np
except Exception:
    np = None

# Paste from optimize_power_model.py output:
OFFSET = {'SOFT': -1.0387912967, 'MEDIUM': 0.0, 'HARD': 0.8279298538}
BASE_DEG = {'SOFT': 0.7708620013, 'MEDIUM': 0.3912977471, 'HARD': 0.1910872173}
CLIFF = {'SOFT': 9.982513, 'MEDIUM': 20.021992, 'HARD': 29.862884}
TEMP_COEFF = 0.0427482786
DEG_EXP = 1.0115024053

COMPOUND_TO_IDX = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}

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

    if np is not None:
        offsets = np.array([OFFSET["SOFT"], 0.0, OFFSET["HARD"]], dtype=np.float64)
        base_deg = np.array([BASE_DEG["SOFT"], BASE_DEG["MEDIUM"], BASE_DEG["HARD"]], dtype=np.float64)
        cliffs = np.array([CLIFF["SOFT"], CLIFF["MEDIUM"], CLIFF["HARD"]], dtype=np.float64)
        eff_deg = base_deg * (1.0 + TEMP_COEFF * temp)
    else:
        offsets = [OFFSET["SOFT"], 0.0, OFFSET["HARD"]]
        base_deg = [BASE_DEG["SOFT"], BASE_DEG["MEDIUM"], BASE_DEG["HARD"]]
        cliffs = [CLIFF["SOFT"], CLIFF["MEDIUM"], CLIFF["HARD"]]
        eff_deg = [d * (1.0 + TEMP_COEFF * temp) for d in base_deg]

    results = []

    for grid in range(1, 21):
        strat = test_case["strategies"][f"pos{grid}"]
        did = strat["driver_id"]

        stints, nstops = build_stints(strat, total_laps)

        maxL = max(L for _, L in stints if L > 0)

        if np is not None:
            ages = np.arange(1, maxL + 1, dtype=np.float64)
            prefix = np.zeros((3, maxL), dtype=np.float64)
            for comp in range(3):
                f = np.maximum(0.0, ages - cliffs[comp]) ** DEG_EXP
                prefix[comp] = np.cumsum(f)
        else:
            prefix = [[0.0] * (maxL + 1) for _ in range(3)]
            for comp in range(3):
                total = 0.0
                cliff = cliffs[comp]
                for age in range(1, maxL + 1):
                    lp = age - cliff
                    if lp > 0.0:
                        total += lp ** DEG_EXP
                    prefix[comp][age] = total

        tt = nstops * pit
        for comp, L in stints:
            if L <= 0:
                continue
            if np is not None:
                tt += L * (base + offsets[comp]) + eff_deg[comp] * prefix[comp, L - 1]
            else:
                tt += L * (base + offsets[comp]) + eff_deg[comp] * prefix[comp][L]

        # Use driver_id only as deterministic tie-break for equal total times.
        results.append((tt, did))

    results.sort()
    return [r[1] for r in results]

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
    except Exception as exc:
        sys.stderr.write(f"race_simulator error: {exc}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()