#!/usr/bin/env python
import json, sys

try:
    import numpy as np
except Exception:
    np = None

# Paste from optimize_power_model.py output:
OFFSET = {'SOFT': -1.0387912967, 'MEDIUM': 0.0, 'HARD': 0.8232843342503486}
BASE_DEG = {'SOFT': 0.7708620013, 'MEDIUM': 0.39195342015625007, 'HARD': 0.19213575169747554}
CLIFF = {'SOFT': 9.982513, 'MEDIUM': 20.010639644561927, 'HARD': 29.862884}
TEMP_OFFSET_COEFF = {
    'SOFT': -0.00016469619917818998,
    'MEDIUM': -0.0034208415942521752,
    'HARD': -0.004159703123025948,
}
TEMP_COEFF = 0.04323067877865523
POWER_DEG_EXP = 1.0115024053

TRACK_PARAMS = {
    "Bahrain": {"deg_mult": [0.991779715753936, 1.0120469812204258, 0.9505235034563455], "temp_scale": 1.1483562997959331, "use_power": False},
    "COTA": {"deg_mult": [1.0, 0.9318484542697919, 1.0], "temp_scale": 1.0, "use_power": False},
    "Monaco": {"deg_mult": [1.0, 1.0, 1.0467060401962263], "temp_scale": 1.0, "use_power": False},
    "Monza": {"deg_mult": [0.9992535024437571, 1.0, 1.0], "temp_scale": 1.0416472937513293, "use_power": True},
    "Silverstone": {"deg_mult": [1.0, 0.9767690814792377, 1.0], "temp_scale": 0.9862000934173344, "use_power": False},
    "Spa": {"deg_mult": [1.0, 1.0284056690142127, 1.0], "temp_scale": 1.0, "use_power": False},
    "Suzuka": {"deg_mult": [1.0, 1.0, 1.0], "temp_scale": 1.0, "use_power": False},
}

START_TIRE_BIAS = {
    "Bahrain": {
        ("D001", "MEDIUM"): -0.016168499432567214,
        ("D002", "MEDIUM"): -0.00045389124080613685,
        ("D003", "SOFT"): -0.030305157420886544,
        ("D004", "SOFT"): 0.006418398347907729,
        ("D009", "HARD"): 0.054391893318645454,
        ("D011", "HARD"): -0.03829109178596499,
        ("D013", "MEDIUM"): -0.046565733751689015,
        ("D014", "SOFT"): -0.03715196254104648,
        ("D014", "MEDIUM"): -0.011796367883454506,
        ("D015", "SOFT"): 0.007976127861935654,
        ("D018", "SOFT"): -0.013788236503463704,
        ("D018", "MEDIUM"): 0.03517706034464649,
        ("D020", "HARD"): -0.0338232758317228,
    },
    "COTA": {
        ("D001", "SOFT"): 0.03635583722536073,
        ("D002", "SOFT"): 0.008066858918819773,
        ("D003", "SOFT"): -0.0040535281222692995,
        ("D004", "SOFT"): -0.007915740883312747,
        ("D005", "SOFT"): 0.02837165090968284,
        ("D006", "SOFT"): -0.0031895041808205557,
        ("D007", "SOFT"): -0.04365785453058786,
        ("D013", "SOFT"): 0.015851260480353,
        ("D014", "SOFT"): 0.045210688690960385,
        ("D015", "SOFT"): -0.06914435158614275,
        ("D016", "SOFT"): 0.060748692335090754,
        ("D018", "SOFT"): 0.07154619284642162,
        ("D019", "SOFT"): 0.042490826845069926,
        ("D019", "MEDIUM"): -0.03164345895199984,
        ("D020", "SOFT"): 0.07092840342608951,
    },
    "Monaco": {
        ("D007", "HARD"): 0.04254455590911144,
        ("D009", "MEDIUM"): -0.025466426271784606,
        ("D017", "MEDIUM"): -0.043226904242324865,
        ("D017", "HARD"): -0.005614541290240541,
        ("D018", "MEDIUM"): -0.019772703217428213,
    },
    "Monza": {
        ("D001", "SOFT"): -0.004654248919196011,
        ("D001", "MEDIUM"): -0.013119467801872335,
        ("D001", "HARD"): -0.08541950372305591,
        ("D002", "SOFT"): -0.011072620145093595,
        ("D002", "HARD"): -0.040430639461890995,
        ("D004", "MEDIUM"): -0.09954361327675187,
        ("D004", "HARD"): -0.049130395703568655,
        ("D005", "MEDIUM"): -0.04249707616456595,
        ("D005", "HARD"): -0.05400108566230949,
        ("D006", "MEDIUM"): -0.013387074868718152,
        ("D007", "MEDIUM"): -0.015417039002637682,
        ("D007", "HARD"): -0.047661938424314945,
        ("D008", "MEDIUM"): -0.0039547904849167245,
        ("D009", "HARD"): -0.0018809183771961446,
        ("D010", "MEDIUM"): -0.04494618097572331,
        ("D012", "MEDIUM"): -0.07317262322739361,
        ("D012", "HARD"): -0.027330814828918096,
        ("D013", "MEDIUM"): 0.02464245414275614,
        ("D013", "HARD"): 0.008762909100139805,
        ("D014", "HARD"): -0.049218964700457994,
        ("D015", "SOFT"): 0.013463462851574987,
        ("D015", "HARD"): 0.025127003709156995,
        ("D016", "SOFT"): 0.030672412462064723,
        ("D016", "MEDIUM"): -0.012418060535721153,
        ("D017", "SOFT"): 0.02271551852907984,
        ("D018", "SOFT"): 6.553164643349763e-05,
        ("D018", "MEDIUM"): 0.011258355502063909,
        ("D019", "SOFT"): 0.007253794497019912,
        ("D019", "MEDIUM"): 0.03631875530320214,
        ("D020", "MEDIUM"): -0.060243038876735,
        ("D020", "HARD"): 0.01210697597078745,
    },
    "Suzuka": {
        ("D003", "MEDIUM"): -0.04966192133209938,
        ("D004", "MEDIUM"): 0.02039273244332057,
        ("D007", "MEDIUM"): 0.04101904872631505,
        ("D017", "SOFT"): -0.04052542894204789,
        ("D017", "MEDIUM"): -0.033508751029168546,
        ("D019", "MEDIUM"): -0.0381731560146179,
    },
}

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
    track = cfg.get("track", "")
    track_cfg = TRACK_PARAMS.get(track, {"deg_mult": [1.0, 1.0, 1.0], "temp_scale": 1.0, "use_power": False})
    start_tire_bias = START_TIRE_BIAS.get(track, {})
    total_laps = int(cfg["total_laps"])
    base = float(cfg["base_lap_time"])
    pit = float(cfg["pit_lane_time"])
    temp = float(cfg["track_temp"])
    dt = temp - 30.0
    use_power_cliff = bool(track_cfg["use_power"])
    deg_mult = track_cfg["deg_mult"]
    temp_scale = float(track_cfg["temp_scale"])

    if np is not None:
        offsets = np.array([
            OFFSET["SOFT"] + TEMP_OFFSET_COEFF["SOFT"] * dt,
            0.0 + TEMP_OFFSET_COEFF["MEDIUM"] * dt,
            OFFSET["HARD"] + TEMP_OFFSET_COEFF["HARD"] * dt,
        ], dtype=np.float64)
        base_deg = np.array([
            BASE_DEG["SOFT"] * deg_mult[0],
            BASE_DEG["MEDIUM"] * deg_mult[1],
            BASE_DEG["HARD"] * deg_mult[2],
        ], dtype=np.float64)
        cliffs = np.array([CLIFF["SOFT"], CLIFF["MEDIUM"], CLIFF["HARD"]], dtype=np.float64)
        eff_deg = base_deg * (1.0 + (TEMP_COEFF * temp_scale) * temp)
    else:
        offsets = [
            OFFSET["SOFT"] + TEMP_OFFSET_COEFF["SOFT"] * dt,
            0.0 + TEMP_OFFSET_COEFF["MEDIUM"] * dt,
            OFFSET["HARD"] + TEMP_OFFSET_COEFF["HARD"] * dt,
        ]
        base_deg = [
            BASE_DEG["SOFT"] * deg_mult[0],
            BASE_DEG["MEDIUM"] * deg_mult[1],
            BASE_DEG["HARD"] * deg_mult[2],
        ]
        cliffs = [CLIFF["SOFT"], CLIFF["MEDIUM"], CLIFF["HARD"]]
        eff_deg = [d * (1.0 + (TEMP_COEFF * temp_scale) * temp) for d in base_deg]

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
                f = np.maximum(0.0, ages - cliffs[comp])
                if use_power_cliff:
                    f = f ** POWER_DEG_EXP
                prefix[comp] = np.cumsum(f)
        else:
            prefix = [[0.0] * (maxL + 1) for _ in range(3)]
            for comp in range(3):
                total = 0.0
                cliff = cliffs[comp]
                for age in range(1, maxL + 1):
                    lp = age - cliff
                    if lp > 0.0:
                        total += (lp ** POWER_DEG_EXP) if use_power_cliff else lp
                    prefix[comp][age] = total

        tt = nstops * pit + start_tire_bias.get((did, strat["starting_tire"]), 0.0)
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
            "finishing_positions": simulate_race(test_case),
        }
        sys.stdout.write(json.dumps(out) + "\n")
    except Exception as exc:
        sys.stderr.write(f"race_simulator error: {exc}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()