#!/usr/bin/env python3
import json
import math
import sys
from typing import Dict, List, Tuple

# These constants are intentionally model-only and do not depend on answer files.
TIRE_TIME_OFFSET = {
    "SOFT": -1.8126154644044687,
    "MEDIUM": 0.0,
    "HARD": 1.504054114548312,
}

TIRE_DEGRADATION = {
    "SOFT": 0.06801611481248367,
    "MEDIUM": 0.007554178348732225,
    "HARD": 0.008897672031848679,
}

TRACK_OFFSET_MULTIPLIER = {
    "Bahrain": 0.8023282356311059,
    "COTA": 0.7322494289085136,
    "Monaco": 0.9799182550081308,
    "Monza": 0.8701363337230563,
    "Silverstone": 1.2392531478165016,
    "Spa": 0.6840722544053337,
    "Suzuka": 0.8782381874400454,
}

TRACK_DEGRADATION_MULTIPLIER = {
    "Bahrain": 0.7577088030596133,
    "COTA": 0.6336101065120995,
    "Monaco": 1.0197531789417136,
    "Monza": 1.1406535156419215,
    "Silverstone": 0.7558291815498017,
    "Spa": 1.3615636988095285,
    "Suzuka": 0.9203351370212807,
}

TIRE_DEGRADATION_THRESHOLD = {
    "SOFT": 5,
    "MEDIUM": 17,
    "HARD": 24,
}

AGE_EXPONENT = 1.3352290903032271
TEMP_SENSITIVITY = 0.009898467395529778

PAIRWISE_WEIGHTS = {
    "physics_total": -0.6264796837501315,
    "start_pos": 0.00638875197785297,
    "pit_count": -0.09379762063009203,
    "soft_laps": -0.10079601719599104,
    "med_laps": 0.09670078057875812,
    "hard_laps": 0.25533776704079286,
    "max_stint": 0.03315504319289965,
    "first_pit": 0.0452132997648109,
    "temp": -0.08441000489740179,
    "track_idx": -0.016313131751044114,
}

ENSEMBLE_WEIGHTS = {
    "wins": 0.9876060510177163,
    "physics_rank": 0.4173795111453445,
    "start_pos": -0.008608363107552332,
    "pit_count": -0.05990773013471952,
    "max_stint": -0.048449505451034505,
    "soft_laps": -0.03480353232378119,
    "hard_laps": 0.05295720885572629,
    "first_pit": 0.0776121196450282,
}

TRACK_INDEX = {
    "Bahrain": 0.0,
    "COTA": 1.0,
    "Monaco": 2.0,
    "Monza": 3.0,
    "Silverstone": 4.0,
    "Spa": 5.0,
    "Suzuka": 6.0,
}


def load_input() -> dict:
    return json.load(sys.stdin)


def build_driver_order(strategies: Dict[str, dict]) -> List[Tuple[str, dict]]:
    # Stable ordering by starting grid key keeps deterministic tie behavior.
    keyed = sorted(
        strategies.items(),
        key=lambda kv: int(kv[0].replace("pos", "")) if kv[0].startswith("pos") else 999,
    )
    return [(entry["driver_id"], entry) for _, entry in keyed]


def build_stints(strategy: dict, total_laps: int) -> List[Tuple[str, int]]:
    current_tire = strategy["starting_tire"]
    pit_stops = sorted(strategy.get("pit_stops", []), key=lambda p: int(p["lap"]))

    prev_lap = 0
    stints: List[Tuple[str, int]] = []
    for stop in pit_stops:
        lap = int(stop["lap"])
        stint_len = lap - prev_lap
        if stint_len > 0:
            stints.append((current_tire, stint_len))
        current_tire = stop["to_tire"]
        prev_lap = lap

    final_stint = total_laps - prev_lap
    if final_stint > 0:
        stints.append((current_tire, final_stint))

    return stints


def power_sum(max_n: int, threshold: int) -> List[float]:
    # Prefix sums of max(0, age-threshold)^p.
    out = [0.0] * (max_n + 1)
    for i in range(1, max_n + 1):
        age_component = i - threshold
        if age_component > 0:
            out[i] = out[i - 1] + (age_component**AGE_EXPONENT)
        else:
            out[i] = out[i - 1]
    return out


def pairwise_probability(left: dict, right: dict) -> float:
    z = 0.0
    for key, weight in PAIRWISE_WEIGHTS.items():
        z += weight * (left[key] - right[key])

    # Stable sigmoid for P(left beats right).
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def rank_from_values(values: List[float]) -> List[float]:
    idx = list(range(len(values)))
    idx.sort(key=lambda i: values[i])
    rank = [0.0] * len(values)
    for r, i in enumerate(idx):
        rank[i] = float(r)
    return rank


def simulate_race(test_case: dict) -> List[str]:
    race_config = test_case["race_config"]
    track = race_config["track"]
    total_laps = int(race_config["total_laps"])
    base_lap_time = float(race_config["base_lap_time"])
    pit_lane_time = float(race_config["pit_lane_time"])
    track_temp = float(race_config["track_temp"])
    temp_mult = 1.0 + (track_temp - 30.0) * TEMP_SENSITIVITY

    offset_mult = TRACK_OFFSET_MULTIPLIER.get(track, 1.0)
    deg_mult = TRACK_DEGRADATION_MULTIPLIER.get(track, 1.0)

    age_power_prefix = {
        tire: power_sum(total_laps, threshold)
        for tire, threshold in TIRE_DEGRADATION_THRESHOLD.items()
    }

    drivers = build_driver_order(test_case["strategies"])

    features: List[dict] = []
    for start_pos, (driver_id, strategy) in enumerate(drivers, start=1):
        stints = build_stints(strategy, total_laps)
        total_time = 0.0
        soft_laps = 0.0
        med_laps = 0.0
        hard_laps = 0.0
        max_stint = 0.0

        pit_laps = [int(p["lap"]) for p in strategy.get("pit_stops", [])]
        first_pit = float(min(pit_laps)) if pit_laps else 0.0

        for tire, stint_len in stints:
            total_time += base_lap_time * stint_len
            total_time += TIRE_TIME_OFFSET[tire] * offset_mult * stint_len
            total_time += TIRE_DEGRADATION[tire] * deg_mult * age_power_prefix[tire][stint_len] * temp_mult

            max_stint = max(max_stint, float(stint_len))
            if tire == "SOFT":
                soft_laps += float(stint_len)
            elif tire == "MEDIUM":
                med_laps += float(stint_len)
            else:
                hard_laps += float(stint_len)

        pit_count = float(len(strategy.get("pit_stops", [])))
        total_time += pit_lane_time * pit_count

        features.append(
            {
                "driver_id": driver_id,
                "start_pos": float(start_pos),
                "physics_total": total_time,
                "pit_count": pit_count,
                "soft_laps": soft_laps,
                "med_laps": med_laps,
                "hard_laps": hard_laps,
                "max_stint": max_stint,
                "first_pit": first_pit,
                "temp": track_temp,
                "track_idx": TRACK_INDEX.get(track, 0.0),
            }
        )

    # Pairwise Borda aggregation plus a light ensemble with physics rank.
    n = len(features)
    wins = [0.0] * n
    for i in range(n):
        for j in range(i + 1, n):
            p = pairwise_probability(features[i], features[j])
            wins[i] += p
            wins[j] += 1.0 - p

    physics_rank = rank_from_values([f["physics_total"] for f in features])

    combined = []
    for i, f in enumerate(features):
        score = 0.0
        score += ENSEMBLE_WEIGHTS["wins"] * (-wins[i])
        score += ENSEMBLE_WEIGHTS["physics_rank"] * physics_rank[i]
        score += ENSEMBLE_WEIGHTS["start_pos"] * f["start_pos"]
        score += ENSEMBLE_WEIGHTS["pit_count"] * f["pit_count"]
        score += ENSEMBLE_WEIGHTS["max_stint"] * f["max_stint"]
        score += ENSEMBLE_WEIGHTS["soft_laps"] * f["soft_laps"]
        score += ENSEMBLE_WEIGHTS["hard_laps"] * f["hard_laps"]
        score += ENSEMBLE_WEIGHTS["first_pit"] * f["first_pit"]
        combined.append((score, f["start_pos"], i))

    combined.sort(key=lambda row: (row[0], row[1]))
    order = [idx for _, _, idx in combined]
    return [features[idx]["driver_id"] for idx in order]


def main() -> None:
    test_case = load_input()
    race_id = test_case.get("race_id", "")
    finishing_positions = simulate_race(test_case)

    output = {
        "race_id": race_id,
        "finishing_positions": finishing_positions,
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
