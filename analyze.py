
import json
import sys

def analyze():
    with open('data/historical_races/races_00000-00999.json') as f:
        races = json.load(f)
    for race in races:
        strats = {}
        found_any = False
        for pos_key, data in race['strategies'].items():
            pit_stops = tuple(sorted([(p['lap'], p['from_tire'], p['to_tire']) for p in data.get('pit_stops', [])]))
            s = (data['starting_tire'], pit_stops)
            if s in strats:
                strats[s].append((pos_key, data['driver_id']))
                found_any = True
            else:
                strats[s] = [(pos_key, data['driver_id'])]
        
        if found_any:
            print(f"Race {race['race_id']}")
            finishing = race['finishing_positions']
            for s, drivers in strats.items():
                if len(drivers) > 1:
                    print(f"  Strategy {s} shared by: {drivers}")
                    indices = sorted([finishing.index(d[1]) for d in drivers])
                    print(f"  Finishing indices: {indices}")
                    if indices[-1] - indices[0] >= len(drivers):
                        print(f"  !!! SOMEONE IS BETWEEN THEM !!!")
            print("-" * 20)
            return # Just check one race for now

if __name__ == '__main__':
    analyze()
