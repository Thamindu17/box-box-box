#!/bin/bash

# Box Box Box - Test Runner
# Usage: ./test_runner.sh "your_solution_command"
# Example: ./test_runner.sh "python solution.py"

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TEST_CASES_DIR="data/test_cases/inputs"
EXPECTED_OUTPUTS_DIR="data/test_cases/expected_outputs"
RUN_COMMAND_FILE="solution/run_command.txt"

# Read solution command from file
if [ ! -f "$RUN_COMMAND_FILE" ]; then
    echo -e "${RED}Error: Run command file not found: $RUN_COMMAND_FILE${NC}"
    echo "Please create $RUN_COMMAND_FILE with your run command"
    echo "Example: python solution/race_simulator.py"
    exit 1
fi

SOLUTION_CMD=$(cat "$RUN_COMMAND_FILE" | tr -d '\r\n')

# Check if test cases exist
if [ ! -d "$TEST_CASES_DIR" ]; then
    echo -e "${RED}Error: Test cases directory not found: $TEST_CASES_DIR${NC}"
    exit 1
fi

# Count test files
TEST_FILES=($(ls $TEST_CASES_DIR/test_*.json 2>/dev/null | sort))
TOTAL_TESTS=${#TEST_FILES[@]}

if [ $TOTAL_TESTS -eq 0 ]; then
    echo -e "${RED}Error: No test files found in $TEST_CASES_DIR${NC}"
    exit 1
fi

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Box Box Box - Test Runner                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Solution Command: ${YELLOW}$SOLUTION_CMD${NC}"
echo -e "Test Cases Found: ${YELLOW}$TOTAL_TESTS${NC}"
echo ""

# Initialize counters
PASSED=0
FAILED=0
ERRORS=0

# Create temp directory for outputs
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

echo -e "${BLUE}Running tests...${NC}"
echo ""

# jq is optional; fall back to Python for JSON parsing when jq is unavailable.
HAS_JQ=false
if command -v jq >/dev/null 2>&1; then
    HAS_JQ=true
fi

json_is_valid() {
    local file="$1"
    if [ "$HAS_JQ" = true ]; then
        jq empty "$file" >/dev/null 2>&1
        return $?
    fi

    python - "$file" <<'PY' >/dev/null 2>&1
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    json.load(f)
PY
}

extract_positions() {
    local file="$1"
    if [ "$HAS_JQ" = true ]; then
        jq -r '.finishing_positions | join(",")' "$file" 2>/dev/null
        return
    fi

    python - "$file" <<'PY' 2>/dev/null
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

positions = data.get("finishing_positions")
if isinstance(positions, list):
    print(",".join(str(x) for x in positions))
else:
    print("null")
PY
}

# Check if we have expected outputs (for local testing)
HAS_ANSWERS=false
if [ -d "$EXPECTED_OUTPUTS_DIR" ]; then
    HAS_ANSWERS=true
fi

# Run tests
for TEST_FILE in "${TEST_FILES[@]}"; do
    TEST_NAME=$(basename "$TEST_FILE" .json)
    TEST_ID=$(echo "$TEST_NAME" | sed 's/test_/TEST_/')

    # Run solution
    OUTPUT_FILE="$TMP_DIR/${TEST_NAME}_output.json"
    ERROR_FILE="$TMP_DIR/${TEST_NAME}_error.log"

    if cat "$TEST_FILE" | eval "$SOLUTION_CMD" > "$OUTPUT_FILE" 2> "$ERROR_FILE"; then
        # Check if output is valid JSON
        if json_is_valid "$OUTPUT_FILE"; then
            # Extract finishing positions from output
            PREDICTED=$(extract_positions "$OUTPUT_FILE")

            if [ -z "$PREDICTED" ] || [ "$PREDICTED" == "null" ]; then
                echo -e "${RED}✗${NC} $TEST_ID - Invalid output format"
                FAILED=$((FAILED + 1))
            elif [ "$HAS_ANSWERS" = true ]; then
                # Compare with expected output if we have answers
                ANSWER_FILE="$EXPECTED_OUTPUTS_DIR/${TEST_NAME}.json"
                if [ -f "$ANSWER_FILE" ]; then
                    EXPECTED=$(extract_positions "$ANSWER_FILE")

                    if [ "$PREDICTED" == "$EXPECTED" ]; then
                        echo -e "${GREEN}✓${NC} $TEST_ID"
                        PASSED=$((PASSED + 1))
                    else
                        echo -e "${RED}✗${NC} $TEST_ID - Incorrect prediction"
                        FAILED=$((FAILED + 1))
                    fi
                else
                    # No answer file for this test
                    echo -e "${YELLOW}?${NC} $TEST_ID - Output generated (no answer file found)"
                    PASSED=$((PASSED + 1))
                fi
            else
                # No answer key, just check format
                echo -e "${YELLOW}?${NC} $TEST_ID - Output generated (no answer key to verify)"
                PASSED=$((PASSED + 1))
            fi
        else
            echo -e "${RED}✗${NC} $TEST_ID - Invalid JSON output"
            FAILED=$((FAILED + 1))
        fi
    else
        # Execution error
        echo -e "${RED}✗${NC} $TEST_ID - Execution error"
        if [ -s "$ERROR_FILE" ]; then
            echo -e "  ${RED}Error:${NC} $(head -n 1 "$ERROR_FILE")"
        fi
        ERRORS=$((ERRORS + 1))
    fi
done

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Results                             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Calculate stats
PASS_RATE=0
if [ $TOTAL_TESTS -gt 0 ]; then
    if command -v bc >/dev/null 2>&1; then
        PASS_RATE=$(echo "scale=1; $PASSED * 100 / $TOTAL_TESTS" | bc)
    else
        PASS_RATE=$(python - "$PASSED" "$TOTAL_TESTS" <<'PY'
import sys

passed = int(sys.argv[1])
total = int(sys.argv[2])
print(f"{(passed * 100.0 / total):.1f}")
PY
)
    fi
fi

echo -e "Total Tests:    ${YELLOW}$TOTAL_TESTS${NC}"
echo -e "Passed:         ${GREEN}$PASSED${NC}"
echo -e "Failed:         ${RED}$FAILED${NC}"
if [ $ERRORS -gt 0 ]; then
    echo -e "Errors:         ${RED}$ERRORS${NC}"
fi
echo ""
echo -e "Pass Rate:      ${GREEN}$PASS_RATE%${NC}"
echo ""

# Final message
if [ "$HAS_ANSWERS" = false ]; then
    echo -e "${YELLOW}Note: Running without expected outputs. Only checking output format.${NC}"
    echo ""
fi

# Exit with appropriate code
if [ $PASSED -eq $TOTAL_TESTS ]; then
    echo -e "${GREEN}🏆 Perfect score! All tests passed!${NC}"
    exit 0
elif [ $PASSED -gt 0 ]; then
    echo -e "${YELLOW}Keep improving! Check failed test cases.${NC}"
    exit 0
else
    echo -e "${RED}No tests passed. Review your implementation.${NC}"
    exit 1
fi
