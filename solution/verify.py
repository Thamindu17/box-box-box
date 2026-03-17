#!/usr/bin/env python3
"""
solution/verify.py
Run: python solution/verify.py
"""

import json
import os
import subprocess
import sys

def main():
    correct = 0
    wrong_list = []
    
    for i in range(1, 101):
        fn = f"test_{i:03d}.json"
        inp = f"data/test_cases/inputs/{fn}"
        out = f"data/test_cases/expected_outputs/{fn}"
        
        with open(inp) as f:
            input_data = f.read()
        with open(out) as f:
            expected = json.load(f)
        
        result = subprocess.run(
            [sys.executable, "solution/race_simulator.py"],
            input=input_data, capture_output=True, text=True
        )
        
        try:
            predicted = json.loads(result.stdout.strip())
            if predicted["finishing_positions"] == expected["finishing_positions"]:
                correct += 1
            else:
                wrong_list.append((i, predicted["finishing_positions"], expected["finishing_positions"]))
        except:
            wrong_list.append((i, None, expected["finishing_positions"]))
    
    # Show first 5 wrong
    for idx, pred, exp in wrong_list[:5]:
        print(f"WRONG test_{idx:03d}:")
        if pred is None:
            print("  ERROR: No valid output")
        else:
            for j, (p, e) in enumerate(zip(pred, exp)):
                if p != e:
                    print(f"  pos {j+1}: predicted={p}, expected={e}")
                    break
    
    print(f"\n{'='*40}")
    print(f"Results: {correct}/100 correct")
    print(f"{'='*40}")

if __name__ == "__main__":
    main()