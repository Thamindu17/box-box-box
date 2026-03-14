
import json
import os
import numpy as np
from scipy.optimize import differential_evolution

def load_test_cases():
    test_cases = []
    inputs_dir = "data/test_cases/inputs"
    outputs_dir = "data/test_cases/expected_outputs"
    
    for i in range(1, 101):
        filename = f"test_{i:03d}.json"
        input_path = os.path.join(inputs_dir, filename)
        output_path = os.path.join(outputs_dir, filename)
        
        with open(input_path, 'r') as f:
            test_input = json.load(f)
        with open(output_path, 'r') as f:
            test_output = json.load(f)
            
        test_cases.append((test_input, test_output["finishing_positions"]))
    return test_cases

def simulate_race(params, test_case):
    (offset_soft, offset_hard, 
     deg_soft, deg_medium, deg_hard, 
     thresh_soft, thresh_medium, thresh_hard, 
     sensitivity, exponent) = params
    
    offsets = {"SOFT": offset_soft, "MEDIUM": 0.0, "HARD": offset_hard}
    degs = {"SOFT": deg_soft, "MEDIUM": deg_medium, "HARD": deg_hard}
    thresholds = {"SOFT": thresh_soft, "MEDIUM": thresh_medium, "HARD": thresh_hard}
    
    race_config = test_case["race_config"]
    total_laps = int(race_config["total_laps"])
    base_lap_time = float(race_config["base_lap_time"])
    pit_lane_time = float(race_config["pit_lane_time"])
    track_temp = float(race_config["track_temp"])
    
    temp_mult = 1.0 + (track_temp - 30.0) * sensitivity

    results = []
    for pos_key, strategy in test_case["strategies"].items():
        driver_id = strategy["driver_id"]
        pit_stops = {p["lap"]: p["to_tire"] for p in strategy.get("pit_stops", [])}
        
        current_tire = strategy["starting_tire"]
        current_tire_age = 0
        total_time = 0.0
        
        for lap in range(1, total_laps + 1):
            current_tire_age += 1
            
            lap_time = base_lap_time + offsets[current_tire]
            
            threshold = thresholds[current_tire]
            if current_tire_age > threshold:
                lap_time += degs[current_tire] * ((current_tire_age - threshold) ** exponent) * temp_mult
            
            total_time += lap_time
            
            if lap in pit_stops:
                total_time += pit_lane_time
                current_tire = pit_stops[lap]
                current_tire_age = 0
        
        pos_num = int(pos_key.replace("pos", ""))
        results.append((total_time, pos_num, driver_id))
    
    results.sort()
    return [r[2] for r in results]

def loss_function(params, test_cases):
    total_loss = 0
    for test_input, expected_positions in test_cases:
        predicted_positions = simulate_race(params, test_input)
        
        # Rank correlation or just sum of squared differences in ranks
        expected_rank = {driver_id: i for i, driver_id in enumerate(expected_positions)}
        for i, driver_id in enumerate(predicted_positions):
            total_loss += (i - expected_rank[driver_id]) ** 2
            
    return total_loss

def main():
    test_cases = load_test_cases()
    
    # Current values as starting point
    # Offset: SOFT: -1.8126154644044687, HARD: 1.504054114548312
    # Degradation: SOFT: 0.06801611481248367, MEDIUM: 0.007554178348732225, HARD: 0.008897672031848679
    # Threshold: SOFT: 5, MEDIUM: 17, HARD: 24
    # Exponent: 1.3352290903032271
    # Sensitivity: 0.009898467395529778
    
    initial_params = [
        -1.8126154644044687, 1.504054114548312,
        0.06801611481248367, 0.007554178348732225, 0.008897672031848679,
        5, 17, 24,
        0.009898467395529778, 1.3352290903032271
    ]
    
    bounds = [
        (-3.0, -1.0), (1.0, 3.0),       # Offsets
        (0.01, 0.2), (0.001, 0.05), (0.001, 0.05), # Degradations
        (1, 10), (10, 25), (15, 35),     # Thresholds
        (0.0, 0.05), (1.0, 2.0)         # Sensitivity, Exponent
    ]
    
    print("Initial loss:", loss_function(initial_params, test_cases))
    
    result = differential_evolution(
        loss_function, 
        bounds, 
        args=(test_cases,), 
        strategy='best2bin',
        maxiter=1000,
        popsize=30,
        tol=0.001,
        mutation=(0.5, 1),
        recombination=0.7,
        workers=-1,
        disp=True
    )
    
    print("Best params:", result.x)
    print("Best loss:", result.fun)
    
    # Check how many pass exactly
    passed = 0
    for test_input, expected_positions in test_cases:
        if simulate_race(result.x, test_input) == expected_positions:
            passed += 1
    print("Passed cases:", passed)

if __name__ == "__main__":
    main()
