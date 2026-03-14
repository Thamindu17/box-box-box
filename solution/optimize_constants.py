#!/usr/bin/env python3
import json
import sys
from scipy.optimize import differential_evolution

def load_races(filename, limit=100):
    try:
        with open(filename, 'r') as f:
            races = json.load(f)
        return races[:limit]
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        sys.exit(1)

def simulate_race_for_opt(race, params):
    # Parameter mapping: 9 parameters total (OFFSET=2, CLIFF=3, BASE_DEG=3, TEMP_MULT=1)
    s_off, h_off, s_cliff, m_cliff, h_cliff, s_deg, m_deg, h_deg, t_mult = params
    
    offsets = {"SOFT": s_off, "MEDIUM": 0.0, "HARD": h_off}
    cliffs = {"SOFT": s_cliff, "MEDIUM": m_cliff, "HARD": h_cliff}
    base_degs = {"SOFT": s_deg, "MEDIUM": m_deg, "HARD": h_deg}

    config = race['race_config']
    total_laps = int(config['total_laps'])
    base_lap_time = float(config['base_lap_time'])
    pit_lane_time = float(config['pit_lane_time'])
    track_temp = float(config['track_temp'])

    driver_results = []
    
    for pos_key, strat in race['strategies'].items():
        driver_id = strat['driver_id']
        pit_map = {int(p['lap']): p['to_tire'] for p in strat.get('pit_stops', [])}
        
        current_tire = strat['starting_tire']
        tire_age = 0
        total_time = 0.0
        
        for lap in range(1, total_laps + 1):
            # Rule: Age increments BEFORE calculation
            tire_age += 1
            
            # Rule: effective_deg = base_deg * (1 + (track_temp - 30) * temp_multiplier)
            effective_deg = base_degs[current_tire] * (1 + (track_temp - 30) * t_mult)
            
            # Rule: Tire Cliff Mechanic
            # Only apply degradation IF the tire age has passed its cliff
            degradation_penalty = 0.0
            if tire_age > cliffs[current_tire]:
                degradation_penalty = (tire_age - cliffs[current_tire]) * effective_deg
            
            # Rule: lap_time calculation
            lap_time = base_lap_time + offsets[current_tire] + degradation_penalty
            total_time += lap_time
            
            # Process pit stop at end of lap
            if lap in pit_map:
                total_time += pit_lane_time
                current_tire = pit_map[lap]
                # Rule: Reset age to 0 so it becomes 1 on next lap
                tire_age = 0
        
        # Tie-breaker: input grid position (pos1 < pos2 < ...)
        pos_num = int(pos_key.replace('pos', ''))
        driver_results.append((total_time, pos_num, driver_id))
        
    # Sort by total time, then grid position
    driver_results.sort()
    return [d[2] for d in driver_results]

def loss_function(params, races):
    total_penalty = 0
    for race in races:
        predicted = simulate_race_for_opt(race, params)
        actual = race['finishing_positions']
        
        # Spearman footrule distance: sum of absolute differences in positions
        for i, driver_id in enumerate(predicted):
            actual_idx = actual.index(driver_id)
            total_penalty += abs(i - actual_idx)
            
    return total_penalty

def main():
    print("Loading 100 historical races for optimization...")
    races = load_races('data/historical_races/races_00000-00999.json', 100)
    
    # Bounds: [SOFT_OFF, HARD_OFF, S_CLIFF, M_CLIFF, H_CLIFF, S_DEG, M_DEG, H_DEG, TEMP_MULT]
    bounds = [
        (-5.0, 0.0),    # SOFT_OFFSET
        (0.0, 5.0),     # HARD_OFFSET
        (1.0, 40.0),    # SOFT_CLIFF
        (1.0, 40.0),    # MEDIUM_CLIFF
        (1.0, 40.0),    # HARD_CLIFF
        (0.0, 0.5),     # SOFT_BASE_DEG
        (0.0, 0.5),     # MEDIUM_BASE_DEG
        (0.0, 0.5),     # HARD_BASE_DEG
        (-0.01, 0.01)   # TEMP_MULTIPLIER
    ]
    
    print("Running Differential Evolution with Tire Cliff model...")
    # Differential Evolution with polish=False to avoid L-BFGS-B failure
    result = differential_evolution(
        loss_function, 
        bounds, 
        args=(races,), 
        strategy='best1bin', 
        maxiter=50, 
        popsize=15, 
        disp=True,
        polish=False
    )
    
    # Unpack and print results
    s_off, h_off, s_cliff, m_cliff, h_cliff, s_deg, m_deg, h_deg, t_mult = result.x
    
    print("\nOptimization Complete!")
    print(f"Success Status: {result.success}")
    print(f"Final Loss Score: {result.fun}")
    print("\n========== OPTIMAL CONSTANTS ==========")
    print(f"OFFSET = {{'SOFT': {s_off:.8f}, 'MEDIUM': 0.0, 'HARD': {h_off:.8f}}}")
    print(f"CLIFF = {{'SOFT': {s_cliff:.4f}, 'MEDIUM': {m_cliff:.4f}, 'HARD': {h_cliff:.4f}}}")
    print(f"BASE_DEGRADATION = {{'SOFT': {s_deg:.8f}, 'MEDIUM': {m_deg:.8f}, 'HARD': {h_deg:.8f}}}")
    print(f"TEMP_MULTIPLIER = {t_mult:.10f}")
    print("=======================================")
    print("Copy these constants into your simulator!")

if __name__ == '__main__':
    main()
