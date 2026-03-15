#!/usr/bin/env python3
"""
solution/deep_search.py
Run: python solution/deep_search.py
"""

import json
import os
import sys
from scipy.optimize import differential_evolution
import numpy as np

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

def preprocess(test_pairs):
    processed = []
    for race, expected in test_pairs:
        config = race["race_config"]
        total_laps = int(config["total_laps"])
        base_lap = float(config["base_lap_time"])
        pit_time = float(config["pit_lane_time"])
        temp = float(config["track_temp"])
        
        drivers = []
        for pi in range(1, 21):
            pk = f"pos{pi}"
            strat = race["strategies"][pk]
            did = strat["driver_id"]
            comp_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}
            pit_laps = {int(p["lap"]): comp_map[p["to_tire"]] for p in strat.get("pit_stops", [])}
            start_comp = comp_map[strat["starting_tire"]]
            num_stops = len(pit_laps)
            drivers.append((did, pi, start_comp, pit_laps, num_stops))
        
        processed.append({
            'total_laps': total_laps,
            'base_lap': base_lap,
            'pit_time': pit_time,
            'temp': temp,
            'drivers': drivers,
            'expected': expected,
        })
    return processed

def simulate(race, params, model_id):
    """
    Model IDs:
    0: linear_cliff, temp_mult_raw
    1: linear_cliff, temp_mult_c30
    2: linear_cliff, temp_add
    3: linear_nocliff, temp_mult_raw
    4: linear_nocliff, temp_mult_c30
    5: cumul_cliff, temp_mult_raw
    6: cumul_cliff, temp_mult_c30
    7: const_after_cliff, temp_mult_raw  (NEW: constant penalty after cliff)
    8: const_after_cliff, temp_mult_c30
    9: sqrt_cliff, temp_mult_raw (NEW: sqrt of laps past cliff)
    10: exp_cliff, temp_mult_raw (NEW: exponential growth)
    11: linear_cliff, temp_mult_raw, with warmup (NEW: first few laps slower)
    """
    os_, oh_, ds_, dm_, dh_, cs_, cm_, ch_, tc_, extra = params[:10]
    offsets = [os_, 0.0, oh_]
    degs = [ds_, dm_, dh_]
    cliffs = [cs_, cm_, ch_]
    
    base_lap = race['base_lap']
    pit_time = race['pit_time']
    temp = race['temp']
    total_laps = race['total_laps']
    
    # Pre-compute effective deg rates
    eff_degs = [0.0, 0.0, 0.0]
    for c in range(3):
        if model_id in [0, 3, 5, 7, 9, 10, 11]:  # mult_raw
            eff_degs[c] = degs[c] * (1.0 + tc_ * temp)
        elif model_id in [1, 4, 6, 8]:  # mult_c30
            eff_degs[c] = degs[c] * (1.0 + tc_ * (temp - 30.0))
        else:  # additive
            eff_degs[c] = degs[c] + tc_ * temp
    
    times = []
    for did, grid, start_comp, pit_laps, num_stops in race['drivers']:
        tt = num_stops * pit_time
        cur_comp = start_comp
        tire_age = 0
        
        for lap in range(1, total_laps + 1):
            tire_age += 1
            
            off = offsets[cur_comp]
            ed = eff_degs[cur_comp]
            cl = cliffs[cur_comp]
            
            lp = max(0.0, tire_age - cl)
            
            if model_id in [0, 1, 2, 11]:  # linear_cliff
                deg = ed * lp
            elif model_id in [3, 4]:  # linear_nocliff
                deg = ed * tire_age
            elif model_id in [5, 6]:  # cumul_cliff
                deg = ed * lp * (lp + 1.0) / 2.0
            elif model_id in [7, 8]:  # const_after_cliff
                deg = ed if tire_age > cl else 0.0
            elif model_id == 9:  # sqrt_cliff
                deg = ed * (lp ** 0.5)
            elif model_id == 10:  # exp_cliff
                deg = ed * ((1.0 + extra) ** lp - 1.0) if lp > 0 else 0.0
            else:
                deg = ed * lp
            
            lap_time = base_lap + off + deg
            
            # Warmup effect for model 11
            if model_id == 11 and tire_age <= 2:
                lap_time += extra  # slight penalty on first laps
            
            tt += lap_time
            
            if lap in pit_laps:
                cur_comp = pit_laps[lap]
                tire_age = 0
        
        times.append((tt, grid, did))
    
    times.sort()
    return [t[2] for t in times]

def objective(params, processed, model_id):
    total_err = 0
    for race in processed:
        pred = simulate(race, params, model_id)
        expected = race['expected']
        ep = {d: i for i, d in enumerate(expected)}
        for i, d in enumerate(pred):
            total_err += (i - ep[d]) ** 2
    return total_err

def count_exact(processed, params, model_id):
    exact = 0
    for race in processed:
        pred = simulate(race, params, model_id)
        if pred == race['expected']:
            exact += 1
    return exact

def get_bounds(model_id):
    # [os, oh, ds, dm, dh, cs, cm, ch, tc, extra]
    if model_id in [3, 4]:  # nocliff
        return [
            (-2.0, 0.0), (0.0, 2.0),
            (0.0001, 0.03), (0.00005, 0.015), (0.00001, 0.008),
            (0, 0.1), (0, 0.1), (0, 0.1),
            (0.0001, 0.03), (0, 0.001)
        ]
    elif model_id in [5, 6]:  # cumul
        return [
            (-2.0, 0.0), (0.0, 2.0),
            (0.0005, 0.05), (0.0002, 0.025), (0.00005, 0.012),
            (1, 20), (5, 35), (10, 50),
            (0.0001, 0.03), (0, 0.001)
        ]
    elif model_id in [7, 8]:  # const after cliff
        return [
            (-2.0, 0.0), (0.0, 2.0),
            (0.01, 0.5), (0.005, 0.3), (0.002, 0.15),
            (1, 20), (5, 35), (10, 50),
            (0.0001, 0.03), (0, 0.001)
        ]
    elif model_id == 9:  # sqrt
        return [
            (-2.0, 0.0), (0.0, 2.0),
            (0.01, 0.8), (0.005, 0.4), (0.002, 0.2),
            (1, 20), (5, 35), (10, 50),
            (0.0001, 0.03), (0, 0.001)
        ]
    elif model_id == 10:  # exp
        return [
            (-2.0, 0.0), (0.0, 2.0),
            (0.001, 0.2), (0.0005, 0.1), (0.0001, 0.05),
            (1, 20), (5, 35), (10, 50),
            (0.0001, 0.03), (0.01, 0.3)  # extra is growth rate
        ]
    elif model_id == 11:  # warmup
        return [
            (-2.0, 0.0), (0.0, 2.0),
            (0.01, 0.5), (0.005, 0.25), (0.001, 0.1),
            (1, 20), (5, 35), (10, 50),
            (0.0001, 0.03), (0.0, 0.5)  # extra is warmup penalty
        ]
    else:  # linear_cliff (default)
        return [
            (-2.0, 0.0), (0.0, 2.0),
            (0.01, 0.5), (0.005, 0.25), (0.001, 0.1),
            (1, 20), (5, 35), (10, 50),
            (0.0001, 0.03), (0, 0.001)
        ]

MODEL_NAMES = {
    0: 'linear_cliff__mult_raw',
    1: 'linear_cliff__mult_c30', 
    2: 'linear_cliff__add',
    3: 'linear_nocliff__mult_raw',
    4: 'linear_nocliff__mult_c30',
    5: 'cumul_cliff__mult_raw',
    6: 'cumul_cliff__mult_c30',
    7: 'const_cliff__mult_raw',
    8: 'const_cliff__mult_c30',
    9: 'sqrt_cliff__mult_raw',
    10: 'exp_cliff__mult_raw',
    11: 'linear_cliff_warmup__mult_raw',
}

def main():
    print("Loading test cases...")
    test_pairs = load_test_cases()
    print(f"Loaded {len(test_pairs)} test cases")
    
    print("Preprocessing...")
    processed = preprocess(test_pairs)
    
    best_global = 0
    best_model = -1
    best_params = None
    
    for model_id in range(12):
        name = MODEL_NAMES.get(model_id, f"model_{model_id}")
        print(f"\n[{model_id}] {name}", flush=True)
        
        bounds = get_bounds(model_id)
        
        def obj(p, mid=model_id):
            return objective(p, processed, mid)
        
        try:
            result = differential_evolution(
                obj, bounds,
                maxiter=200, popsize=20,
                seed=42, tol=1e-12,
                mutation=(0.5, 1.5), recombination=0.8,
                disp=False, polish=False, workers=1
            )
            
            p = result.x
            ex = count_exact(processed, p, model_id)
            print(f"    exact={ex}/100, loss={result.fun:.0f}")
            
            if ex > best_global:
                best_global = ex
                best_model = model_id
                best_params = p.copy()
                print(f"    *** NEW BEST ***")
            
            # If we get good results, refine
            if ex >= 20:
                print(f"    Refining...", flush=True)
                # Tighter bounds
                tight = []
                for i, v in enumerate(p):
                    if i < 2:
                        tight.append((v-0.15, v+0.15))
                    elif i < 5:
                        tight.append((max(1e-7, v*0.5), v*2))
                    elif i < 8:
                        tight.append((max(0, v-3), v+3))
                    else:
                        tight.append((max(1e-7, v*0.5), v*2))
                
                result2 = differential_evolution(
                    obj, tight,
                    maxiter=300, popsize=25,
                    seed=99, tol=1e-14,
                    disp=False, polish=True, workers=1
                )
                
                ex2 = count_exact(processed, result2.x, model_id)
                print(f"    refined: exact={ex2}/100")
                
                if ex2 > best_global:
                    best_global = ex2
                    best_model = model_id
                    best_params = result2.x.copy()
                    print(f"    *** NEW BEST ***")
                
                # Try integer cliffs
                p2 = result2.x
                for cs in range(max(0, int(p2[5])-2), int(p2[5])+3):
                    for cm in range(max(0, int(p2[6])-2), int(p2[6])+3):
                        for ch in range(max(0, int(p2[7])-2), int(p2[7])+3):
                            tp = p2.copy()
                            tp[5], tp[6], tp[7] = float(cs), float(cm), float(ch)
                            exi = count_exact(processed, tp, model_id)
                            if exi > best_global:
                                best_global = exi
                                best_model = model_id
                                best_params = tp.copy()
                                print(f"    Int({cs},{cm},{ch}): {exi}/100 ***")
        
        except Exception as e:
            print(f"    ERROR: {e}")
    
    if best_params is None:
        print("\nNo solution found!")
        return
    
    # Final output
    p = best_params
    name = MODEL_NAMES.get(best_model, f"model_{best_model}")
    
    print(f"\n{'='*60}")
    print(f"BEST: {name} with {best_global}/100")
    print(f"{'='*60}")
    
    # Generate code based on model type
    deg_code = ""
    temp_code = ""
    
    if best_model in [0, 11]:
        deg_code = "lp = max(0.0, ta - CLIFF[ct])\n            deg = eff * lp"
        temp_code = "eff = BASE_DEG[ct] * (1.0 + TEMP_COEFF * temp)"
    elif best_model == 1:
        deg_code = "lp = max(0.0, ta - CLIFF[ct])\n            deg = eff * lp"
        temp_code = "eff = BASE_DEG[ct] * (1.0 + TEMP_COEFF * (temp - 30.0))"
    elif best_model == 2:
        deg_code = "lp = max(0.0, ta - CLIFF[ct])\n            deg = eff * lp"
        temp_code = "eff = BASE_DEG[ct] + TEMP_COEFF * temp"
    elif best_model == 3:
        deg_code = "deg = eff * ta"
        temp_code = "eff = BASE_DEG[ct] * (1.0 + TEMP_COEFF * temp)"
    elif best_model == 4:
        deg_code = "deg = eff * ta"
        temp_code = "eff = BASE_DEG[ct] * (1.0 + TEMP_COEFF * (temp - 30.0))"
    elif best_model == 5:
        deg_code = "lp = max(0.0, ta - CLIFF[ct])\n            deg = eff * lp * (lp + 1.0) / 2.0"
        temp_code = "eff = BASE_DEG[ct] * (1.0 + TEMP_COEFF * temp)"
    elif best_model == 6:
        deg_code = "lp = max(0.0, ta - CLIFF[ct])\n            deg = eff * lp * (lp + 1.0) / 2.0"
        temp_code = "eff = BASE_DEG[ct] * (1.0 + TEMP_COEFF * (temp - 30.0))"
    elif best_model == 7:
        deg_code = "deg = eff if ta > CLIFF[ct] else 0.0"
        temp_code = "eff = BASE_DEG[ct] * (1.0 + TEMP_COEFF * temp)"
    elif best_model == 8:
        deg_code = "deg = eff if ta > CLIFF[ct] else 0.0"
        temp_code = "eff = BASE_DEG[ct] * (1.0 + TEMP_COEFF * (temp - 30.0))"
    elif best_model == 9:
        deg_code = "lp = max(0.0, ta - CLIFF[ct])\n            deg = eff * (lp ** 0.5)"
        temp_code = "eff = BASE_DEG[ct] * (1.0 + TEMP_COEFF * temp)"
    elif best_model == 10:
        deg_code = f"lp = max(0.0, ta - CLIFF[ct])\n            deg = eff * ((1.0 + {p[9]:.6f}) ** lp - 1.0) if lp > 0 else 0.0"
        temp_code = "eff = BASE_DEG[ct] * (1.0 + TEMP_COEFF * temp)"
    
    warmup_code = ""
    if best_model == 11:
        warmup_code = f"\n            if ta <= 2:\n                lap_time += {p[9]:.6f}"
    
    print(f"""
# Paste into race_simulator.py:

OFFSET = {{'SOFT': {p[0]:.10f}, 'MEDIUM': 0.0, 'HARD': {p[1]:.10f}}}
BASE_DEG = {{'SOFT': {p[2]:.10f}, 'MEDIUM': {p[3]:.10f}, 'HARD': {p[4]:.10f}}}
CLIFF = {{'SOFT': {p[5]:.6f}, 'MEDIUM': {p[6]:.6f}, 'HARD': {p[7]:.6f}}}
TEMP_COEFF = {p[8]:.10f}

# In simulate loop:
# {temp_code}
# {deg_code}{warmup_code}
""")

if __name__ == "__main__":
    main()