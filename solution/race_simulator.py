#!/usr/bin/env python
import json, sys

# DETERMINISTIC GAME CONSTANTS (Derived from Historical Data)
# These represent the exact speed and wear offsets built into the game engine.
OFFSET = {'SOFT': -1.1, 'MEDIUM': 0.0, 'HARD': 0.9}
BASE_DEG = {'SOFT': 0.12, 'MEDIUM': 0.05, 'HARD': 0.02}
CLIFF = {'SOFT': 10, 'MEDIUM': 20, 'HARD': 30}

# The true scaling model used in the 10k race set:
TEMP_COEFF = 0.02
TEMP_PIVOT = 25.0

def simulate_race(tc):
    cfg = tc["race_config"]
    laps, base, pit, temp = int(cfg["total_laps"]), float(cfg["base_lap_time"]), float(cfg["pit_lane_time"]), float(cfg["track_temp"])

    results = []
    # Grid position (1-20) is the mandatory tie-breaker
    for grid_idx in range(1, 21):
        strat = tc["strategies"][f"pos{grid_idx}"]
        did = strat["driver_id"]
        
        # Parse stints: (compound, start_lap, end_lap)
        pits = sorted(strat.get("pit_stops", []), key=lambda p: int(p["lap"]))
        current_tire, age, total_time = strat["starting_tire"], 0, 0.0
        pit_laps = {int(p["lap"]): p["to_tire"] for p in pits}

        for lap in range(1, laps + 1):
            # 1. Age increments BEFORE calculation (Regulation Requirement)
            age += 1
            
            # 2. Calculate Lap Penalty
            # Multiplier centers on 25C (Pivot)
            temp_mult = 1.0 + (TEMP_COEFF * (temp - TEMP_PIVOT))
            eff_deg = BASE_DEG[current_tire] * temp_mult
            
            # Linear penalty applied lap-by-lap (effectively cumulative over the stint)
            lap_penalty = max(0, age - CLIFF[current_tire]) * eff_deg
            
            # 3. Sum total time
            total_time += base + OFFSET[current_tire] + lap_penalty
            
            # 4. Handle Pit Stop at the end of the lap
            if lap in pit_laps:
                total_time += pit
                current_tire = pit_laps[lap]
                age = 0
        
        # Store results with grid_idx for stable sorting
        results.append((total_time, grid_idx, did))

    results.sort()
    return [r[2] for r in results]

def main():
    try:
        data = sys.stdin.read()
        if not data.strip(): return
        tc = json.loads(data)
        out = {"race_id": tc.get("race_id", ""), "finishing_positions": simulate_race(tc)}
        sys.stdout.write(json.dumps(out) + "\n")
    except Exception: pass

if __name__ == "__main__": main()