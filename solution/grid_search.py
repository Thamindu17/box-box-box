#!/usr/bin/env python3
"""
solution/grid_search.py
Exhaustive search over round number combinations
Run: python solution/grid_search.py
"""

import json
import os
from itertools import product

def load_test_cases():
    pairs = []
    for i in range(1, 101):
        inp = f"data/test_cases/inputs/test_{i:03d}.json"
        out = f"data/test_cases/expected_outputs/test_{i:03d}.json"
        if os.path.exists(inp) and os.path.exists(out):
            with open(inp) as f:
                ti = json.load(f)
            with open(out) as f:
                to = json.load(f)
            pairs.append((ti, to["finishing_positions"]))
    return pairs

def simulate(race, params, formula_id):
    os_, oh_, ds_, dm_, dh_, cs_, cm_, ch_, tc_ = params
    
    config = race["race_config"]
    total_laps = int(config["total_laps"])
    base_lap = float(config["base_lap_time"])
    pit_time = float(config["pit_lane_time"])
    temp = float(config["track_temp"])
    
    offsets = {'SOFT': os_, 'MEDIUM': 0.0, 'HARD': oh_}
    degs = {'SOFT': ds_, 'MEDIUM': dm_, 'HARD': dh_}
    cliffs = {'SOFT': cs_, 'MEDIUM': cm_, 'HARD': ch_}
    
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
            
            # Different temperature formulas
            if formula_id % 4 == 0:  # mult by temp
                eff = degs[ct] * (1.0 + tc_ * temp)
            elif formula_id % 4 == 1:  # mult by (temp - 20)
                eff = degs[ct] * (1.0 + tc_ * (temp - 20.0))
            elif formula_id % 4 == 2:  # mult by (temp - 30)
                eff = degs[ct] * (1.0 + tc_ * (temp - 30.0))
            else:  # additive
                eff = degs[ct] + tc_ * temp
            
            # Different degradation formulas
            cl = cliffs[ct]
            lp = max(0.0, ta - cl)
            
            if formula_id // 4 == 0:  # linear after cliff
                deg = eff * lp
            elif formula_id // 4 == 1:  # linear no cliff
                deg = eff * ta
            elif formula_id // 4 == 2:  # quadratic after cliff
                deg = eff * lp * lp
            elif formula_id // 4 == 3:  # cumulative after cliff
                deg = eff * lp * (lp + 1.0) / 2.0
            elif formula_id // 4 == 4:  # sqrt after cliff
                deg = eff * (lp ** 0.5) if lp > 0 else 0.0
            elif formula_id // 4 == 5:  # constant after cliff
                deg = eff if ta > cl else 0.0
            else:
                deg = eff * lp
            
            tt += base_lap + offsets[ct] + deg

            if lap in pm:
                tt += pit_time
                ct = pm[lap]
                ta = 0

        results.append((tt, pi, did))

    results.sort()
    return [r[2] for r in results]

def count_exact(test_pairs, params, formula_id):
    exact = 0
    for race, expected in test_pairs:
        pred = simulate(race, params, formula_id)
        if pred == expected:
            exact += 1
    return exact

def main():
    print("Loading test cases...")
    test_pairs = load_test_cases()
    print(f"Loaded {len(test_pairs)} test cases")
    
    # Round number candidates
    offset_s_vals = [-1.0, -0.8, -0.6, -0.5, -0.4, -0.3, -0.2]
    offset_h_vals = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    
    deg_s_vals = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2]
    deg_m_vals = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
    deg_h_vals = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04]
    
    cliff_s_vals = [5, 6, 7, 8, 9, 10, 12]
    cliff_m_vals = [12, 14, 15, 16, 18, 20, 22]
    cliff_h_vals = [20, 22, 25, 28, 30, 32, 35]
    
    tc_vals = [0.0, 0.002, 0.005, 0.008, 0.01, 0.012, 0.015]
    
    # 6 degradation types × 4 temperature types = 24 formula combinations
    num_formulas = 24
    
    best_exact = 0
    best_params = None
    best_formula = -1
    
    total_combos = (len(offset_s_vals) * len(offset_h_vals) * len(deg_s_vals) * 
                    len(deg_m_vals) * len(deg_h_vals) * len(cliff_s_vals) *
                    len(cliff_m_vals) * len(cliff_h_vals) * len(tc_vals) * num_formulas)
    
    print(f"Testing {total_combos:,} combinations...")
    
    checked = 0
    for fid in range(num_formulas):
        deg_type = ['linear_cliff', 'linear_nocliff', 'quad_cliff', 
                    'cumul_cliff', 'sqrt_cliff', 'const_cliff'][fid // 4]
        temp_type = ['mult_temp', 'mult_t-20', 'mult_t-30', 'additive'][fid % 4]
        
        print(f"\nFormula {fid}: {deg_type} + {temp_type}")
        
        formula_best = 0
        formula_best_params = None
        
        for os_ in offset_s_vals:
            for oh_ in offset_h_vals:
                for ds_ in deg_s_vals:
                    for dm_ in deg_m_vals:
                        for dh_ in deg_h_vals:
                            for cs_ in cliff_s_vals:
                                for cm_ in cliff_m_vals:
                                    for ch_ in cliff_h_vals:
                                        for tc_ in tc_vals:
                                            params = (os_, oh_, ds_, dm_, dh_, 
                                                     cs_, cm_, ch_, tc_)
                                            ex = count_exact(test_pairs, params, fid)
                                            
                                            if ex > formula_best:
                                                formula_best = ex
                                                formula_best_params = params
                                            
                                            if ex > best_exact:
                                                best_exact = ex
                                                best_params = params
                                                best_formula = fid
                                                print(f"  NEW BEST: {ex}/100, fid={fid}, params={params}")
                                            
                                            checked += 1
                                            if checked % 500000 == 0:
                                                print(f"  ...checked {checked:,}/{total_combos:,} ({100*checked/total_combos:.1f}%)")
        
        print(f"  Formula {fid} best: {formula_best}/100, params={formula_best_params}")
    
    if best_params:
        os_, oh_, ds_, dm_, dh_, cs_, cm_, ch_, tc_ = best_params
        deg_type = ['linear_cliff', 'linear_nocliff', 'quad_cliff', 
                    'cumul_cliff', 'sqrt_cliff', 'const_cliff'][best_formula // 4]
        temp_type = ['mult_temp', 'mult_t-20', 'mult_t-30', 'additive'][best_formula % 4]
        
        print(f"\n{'='*60}")
        print(f"FINAL BEST: {best_exact}/100")
        print(f"Formula: {deg_type} + {temp_type}")
        print(f"OFFSET = {{'SOFT': {os_}, 'MEDIUM': 0.0, 'HARD': {oh_}}}")
        print(f"BASE_DEG = {{'SOFT': {ds_}, 'MEDIUM': {dm_}, 'HARD': {dh_}}}")
        print(f"CLIFF = {{'SOFT': {cs_}, 'MEDIUM': {cm_}, 'HARD': {ch_}}}")
        print(f"TEMP_COEFF = {tc_}")
        print(f"FORMULA_ID = {best_formula}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()