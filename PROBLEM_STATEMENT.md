
# Box Box Box - Problem Statement

## Objective

Build an F1 race simulator that predicts finishing positions based on tire strategies and race conditions. Given a race configuration and pit stop strategies for 20 drivers, your program must accurately predict the final race order.

---

## The Challenge

You are provided with 30,000 historical F1 races, each containing complete race data and results. Your task is to analyze this data to understand the underlying race mechanics, then build a simulator that can predict outcomes for new races.

---

## Input Specification

Your program receives a JSON object via stdin containing:

### Race Configuration
- `race_id`: Unique identifier for the race
- `race_config`: Race parameters
  - `track`: Track name (e.g., "Monza", "Silverstone")
  - `total_laps`: Number of laps in the race (integer)
  - `base_lap_time`: Track's baseline lap time in seconds (float)
  - `pit_lane_time`: Time penalty for a pit stop in seconds (float)
  - `track_temp`: Track temperature in Celsius (integer)

### Strategies
- `strategies`: Object containing strategies for positions pos1 through pos20
  - `driver_id`: Unique driver identifier (e.g., "D001")
  - `starting_tire`: Initial tire compound ("SOFT", "MEDIUM", or "HARD")
  - `pit_stops`: Array of planned pit stops
    - `lap`: Lap number when pit stop occurs (integer)
    - `from_tire`: Tire compound before pit stop
    - `to_tire`: New tire compound after pit stop

See [data_format.md](docs/data_format.md) for detailed format specification.

---

## Output Specification

Your program must output a JSON object to stdout containing:

- `race_id`: Echo the input race_id
- `finishing_positions`: Array of driver IDs ordered from 1st to 20th place

**Example Output:**
```json
{
  "race_id": "R00001",
  "finishing_positions": ["D007", "D015", "D003", "D012", ..., "D019"]
}
```

The array must contain exactly 20 driver IDs in finishing order.

---

## Race Rules

### Simulation Model
- Each car races independently (no car-to-car interaction)
- Winner is determined by shortest total race time
- Total race time = sum of all lap times + pit stop penalties
- See [regulations.md](docs/regulations.md) for complete racing rules

### Key Factors Affecting Lap Times
1. **Base Lap Time**: Track's characteristic speed
2. **Tire Compound**: Each compound (SOFT/MEDIUM/HARD) has different performance
3. **Tire Degradation**: Tire performance changes with usage
4. **Track Temperature**: Affects tire behavior
5. **Pit Stops**: Time penalty when entering pit lane

### Constraints
- All 20 cars have identical performance capabilities
- All 20 drivers have equal skill levels
- Strategy is the only differentiating factor
- Track conditions remain constant during each race
- No external events (safety cars, accidents, weather changes)
- All 20 drivers must be included in finishing positions

---

## Data Provided

### Historical Race Data
**Location**: `data/historical_races/`

30,000 complete race records organized in files:
- `races_00000-00999.json`
- `races_01000-01999.json`
- ...
- `races_29000-29999.json`

Each record contains:
- Complete race configuration
- All 20 drivers' strategies (starting tires and pit stop plans)
- Actual finishing positions (the ground truth)

**Purpose**: Analyze this data to reverse-engineer the race simulation mechanics.

### Test Cases
**Location**: `data/test_cases/`

100 test cases to validate your solution:
- **Inputs**: `inputs/test_001.json` through `inputs/test_100.json`
- **Expected Outputs**: `expected_outputs/test_001.json` through `expected_outputs/test_100.json`

Your simulator must predict the finishing positions for these test cases.

---

## Implementation Requirements

Your solution must:

1. **Read Input**: Accept race configuration JSON from stdin
2. **Simulate Race**: Process the race lap-by-lap
3. **Calculate Lap Times**: Determine each driver's lap time based on:
   - Tire compound currently in use
   - Number of laps completed on current tires
   - Track conditions
4. **Handle Pit Stops**: Apply time penalties when drivers pit
5. **Determine Results**: Sort drivers by total race time
6. **Output Results**: Write finishing positions JSON to stdout

### Program Interface
```bash
# Your program reads from stdin and writes to stdout
cat data/test_cases/inputs/test_001.json | your_solution_command
# Output: {"race_id": "TEST_001", "finishing_positions": [...]}
```

---

## Submission

1. Place your solution code in the `solution/` directory
2. Update `solution/run_command.txt` with the command to run your program from the repository root
3. Test locally using: `./test_runner.sh`
4. See [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md) for complete submission instructions

---

## Evaluation

### Scoring
- Your solution is tested against 100 test cases
- **Score = (Correct Predictions / 100) × 100%**
- A prediction is correct only if the entire finishing order matches exactly

### Success Metrics
- **Passing**: Correctly predict the majority of test cases
- **Strong Performance**: 80%+ accuracy
- **Excellent Performance**: 95%+ accuracy

---

## Example

### Input (test_001.json)
```json
{
  "race_id": "TEST_001",
  "race_config": {
    "track": "Monza",
    "total_laps": 53,
    "base_lap_time": 82.5,
    "pit_lane_time": 22.0,
    "track_temp": 32
  },
  "strategies": {
    "pos1": {
      "driver_id": "D001",
      "starting_tire": "SOFT",
      "pit_stops": [
        {"lap": 18, "from_tire": "SOFT", "to_tire": "MEDIUM"},
        {"lap": 38, "from_tire": "MEDIUM", "to_tire": "HARD"}
      ]
    },
    "pos2": {
      "driver_id": "D002",
      "starting_tire": "MEDIUM",
      "pit_stops": [
        {"lap": 26, "from_tire": "MEDIUM", "to_tire": "HARD"}
      ]
    }
    // ... pos3 through pos20
  }
}
```

### Output
```json
{
  "race_id": "TEST_001",
  "finishing_positions": [
    "D002", "D001", "D007", "D015", "D012",
    "D008", "D003", "D011", "D019", "D005",
    "D014", "D020", "D004", "D009", "D018",
    "D006", "D013", "D017", "D010", "D016"
  ]
}
```

---

## Technical Approach

### Phase 1: Data Exploration
- Examine historical race records
- Identify patterns in lap times
- Understand how different factors affect performance

### Phase 2: Model Development
- Determine the relationship between tire compound and lap time
- Model tire degradation behavior
- Account for temperature effects
- Build lap time calculation function

### Phase 3: Race Simulation
- Implement lap-by-lap simulation
- Track tire age for each driver
- Handle pit stop timing and tire changes
- Calculate total race times

### Phase 4: Validation
- Test against known historical results
- Validate with provided test cases
- Refine model based on errors

---

## Key Assumptions

- The same race mechanics apply to all 30,000+ races
- All races follow the regulations defined in [regulations.md](docs/regulations.md)
- Historical data is accurate and complete
- Test cases follow the same rules as historical data
- No randomness - outcomes are deterministic

---

## Documentation References

- **[docs/regulations.md](docs/regulations.md)** - Complete F1 racing rules
- **[docs/data_format.md](docs/data_format.md)** - JSON structure specification
- **[docs/faq.md](docs/faq.md)** - Frequently asked questions
- **[SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md)** - How to submit your solution

---

Good luck! 🏁
