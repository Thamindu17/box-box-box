#!/usr/bin/env python3
"""
solution/analyze_historical.py
Analyze historical data to extract formula patterns
Run: python solution/analyze_historical.py
"""

import json
import os
from collections import defaultdict

def load_historical():
    races = []
    fn = "data/historical_races/races_00000-00999.json"
    if os.path.exists(fn):
        with open(fn) as f:
            data = json.load(f)
            if isinstance(data, list):
                races.extend(data)
    return races

def get_strategy_key(strat, total_laps):
    """Create a hashable key for a strategy."""
    stops = tuple(sorted([(int(p['lap']), p['to_tire']) for p in strat.get('pit_stops', [])]))
    return (strat['starting_tire'], stops)

def analyze_same_strategy_pairs(races):
    """Find drivers with identical strategies and verify they're ordered by grid."""
    print("="*60)
    print("IDENTICAL STRATEGY ANALYSIS")
    print("="*60)
    
    all_good = True
    for race in races[:500]:
        total_laps = int(race['race_config']['total_laps'])
        finishing = race['finishing_positions']
        
        strat_groups = defaultdict(list)
        for pk, strat in race['strategies'].items():
            key = get_strategy_key(strat, total_laps)
            grid = int(pk.replace('pos', ''))
            did = strat['driver_id']
            fin_pos = finishing.index(did)
            strat_groups[key].append((grid, fin_pos, did))
        
        for key, drivers in strat_groups.items():
            if len(drivers) > 1:
                # Sort by grid position
                drivers.sort(key=lambda x: x[0])
                # Check if finishing positions are also in order
                fin_positions = [d[1] for d in drivers]
                if fin_positions != sorted(fin_positions):
                    print(f"VIOLATION in {race['race_id']}: {key}")
                    print(f"  Drivers (grid, finish, id): {drivers}")
                    all_good = False
    
    if all_good:
        print("✓ All identical strategies finish in grid order")
    return all_good

def find_simple_comparison_pairs(races):
    """Find pairs where we can isolate single variable effects."""
    print("\n" + "="*60)
    print("SIMPLE COMPARISON PAIRS")
    print("="*60)
    
    # Find no-stop races with different starting compounds
    compound_comparisons = []
    
    for race in races[:1000]:
        total_laps = int(race['race_config']['total_laps'])
        temp = int(race['race_config']['track_temp'])
        base_lap = float(race['race_config']['base_lap_time'])
        finishing = race['finishing_positions']
        
        no_stop_drivers = []
        for pk, strat in race['strategies'].items():
            if len(strat.get('pit_stops', [])) == 0:
                grid = int(pk.replace('pos', ''))
                did = strat['driver_id']
                fin_pos = finishing.index(did)
                compound = strat['starting_tire']
                no_stop_drivers.append((compound, grid, fin_pos, did))
        
        # Find pairs with different compounds
        for i in range(len(no_stop_drivers)):
            for j in range(i+1, len(no_stop_drivers)):
                d1, d2 = no_stop_drivers[i], no_stop_drivers[j]
                if d1[0] != d2[0]:
                    compound_comparisons.append({
                        'race_id': race['race_id'],
                        'total_laps': total_laps,
                        'temp': temp,
                        'base_lap': base_lap,
                        'driver1': d1,
                        'driver2': d2,
                    })
    
    print(f"Found {len(compound_comparisons)} no-stop compound comparison pairs")
    
    # Analyze: when does SOFT beat MEDIUM, MEDIUM beat HARD, etc.?
    wins = defaultdict(lambda: {'wins': 0, 'losses': 0, 'laps': [], 'temps': []})
    
    for comp in compound_comparisons:
        c1, g1, f1, d1 = comp['driver1']
        c2, g2, f2, d2 = comp['driver2']
        
        # Who finished ahead?
        if f1 < f2:
            winner, loser = c1, c2
        elif f2 < f1:
            winner, loser = c2, c1
        else:
            # Same position - shouldn't happen
            continue
        
        key = f"{winner} vs {loser}"
        if f1 < f2:
            wins[key]['wins'] += 1
        else:
            wins[key]['losses'] += 1
        wins[key]['laps'].append(comp['total_laps'])
        wins[key]['temps'].append(comp['temp'])
    
    print("\nCompound matchups (no-stop races):")
    for key, data in sorted(wins.items()):
        avg_laps = sum(data['laps']) / len(data['laps']) if data['laps'] else 0
        avg_temp = sum(data['temps']) / len(data['temps']) if data['temps'] else 0
        print(f"  {key}: wins={data['wins']}, losses={data['losses']}, avg_laps={avg_laps:.1f}, avg_temp={avg_temp:.1f}")
    
    return compound_comparisons

def analyze_pit_stop_effect(races):
    """Analyze how pit stops affect relative positions."""
    print("\n" + "="*60)
    print("PIT STOP ANALYSIS")
    print("="*60)
    
    # Compare drivers with different number of stops
    stop_counts = defaultdict(list)
    
    for race in races[:500]:
        finishing = race['finishing_positions']
        pit_time = float(race['race_config']['pit_lane_time'])
        
        for pk, strat in race['strategies'].items():
            num_stops = len(strat.get('pit_stops', []))
            did = strat['driver_id']
            fin_pos = finishing.index(did)
            stop_counts[num_stops].append(fin_pos)
    
    print("Average finishing position by stop count:")
    for stops in sorted(stop_counts.keys()):
        positions = stop_counts[stops]
        avg = sum(positions) / len(positions)
        print(f"  {stops} stops: avg_pos={avg:.2f}, count={len(positions)}")

def test_specific_formula(races, test_pairs):
    """Test a specific formula hypothesis."""
    print("\n" + "="*60)
    print("TESTING FORMULA HYPOTHESES")
    print("="*60)
    
    # Hypothesis: Simple linear degradation from lap 1, no cliff
    # lap_time = base + offset[c] + deg[c] * age * (1 + tc * temp)
    
    def test_formula(params, temp_model, deg_model):
        os_, oh_, ds_, dm_, dh_, cs_, cm_, ch_, tc_ = params
        
        exact = 0
        for race, expected in test_pairs:
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
                    
                    if temp_model == 'mult':
                        eff = degs[ct] * (1.0 + tc_ * temp)
                    elif temp_model == 'mult30':
                        eff = degs[ct] * (1.0 + tc_ * (temp - 30.0))
                    else:
                        eff = degs[ct] + tc_ * temp
                    
                    cl = cliffs[ct]
                    lp = max(0.0, ta - cl)
                    
                    if deg_model == 'linear':
                        deg = eff * lp
                    elif deg_model == 'nocliff':
                        deg = eff * ta
                    elif deg_model == 'cumul':
                        deg = eff * lp * (lp + 1.0) / 2.0
                    else:
                        deg = eff * lp
                    
                    tt += base_lap + offsets[ct] + deg
                    
                    if lap in pm:
                        tt += pit_time
                        ct = pm[lap]
                        ta = 0

                results.append((tt, pi, did))
            
            results.sort()
            pred = [r[2] for r in results]
            if pred == expected:
                exact += 1
        
        return exact
    
    # Load test cases
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
    
    # Test various round number combinations
    best = 0
    best_info = None
    
    print("Testing round number combinations...")
    
    for os_ in [-0.5, -0.4, -0.3]:
        for oh_ in [0.3, 0.4, 0.5]:
            for ds_ in [0.05, 0.075, 0.1]:
                for dm_ in [0.025, 0.04, 0.05]:
                    for dh_ in [0.01, 0.02, 0.025]:
                        for cs_ in [8, 10, 12]:
                            for cm_ in [16, 18, 20]:
                                for ch_ in [25, 28, 30]:
                                    for tc_ in [0.005, 0.01, 0.015]:
                                        for tm in ['mult', 'mult30', 'add']:
                                            for dm_model in ['linear', 'cumul']:
                                                params = (os_, oh_, ds_, dm_, dh_, cs_, cm_, ch_, tc_)
                                                ex = test_formula(params, tm, dm_model)
                                                if ex > best:
                                                    best = ex
                                                    best_info = (params, tm, dm_model)
                                                    print(f"  {ex}/100: {tm} {dm_model} {params}")
    
    if best_info:
        print(f"\nBest found: {best}/100")
        print(f"  Params: {best_info[0]}")
        print(f"  Temp model: {best_info[1]}")
        print(f"  Deg model: {best_info[2]}")

def main():
    print("Loading data...")
    races = load_historical()
    print(f"Loaded {len(races)} historical races")
    
    # Load test cases for formula testing
    test_pairs = []
    for i in range(1, 101):
        inp = f"data/test_cases/inputs/test_{i:03d}.json"
        out = f"data/test_cases/expected_outputs/test_{i:03d}.json"
        if os.path.exists(inp) and os.path.exists(out):
            with open(inp) as f:
                ti = json.load(f)
            with open(out) as f:
                to = json.load(f)
            test_pairs.append((ti, to["finishing_positions"]))
    
    print(f"Loaded {len(test_pairs)} test cases")
    
    analyze_same_strategy_pairs(races)
    find_simple_comparison_pairs(races)
    analyze_pit_stop_effect(races)
    test_specific_formula(races, test_pairs)

if __name__ == "__main__":
    main()