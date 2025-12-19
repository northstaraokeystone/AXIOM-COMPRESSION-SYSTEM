#!/bin/bash
# CLAUDEME Guardrail: File Size Checker
# HARD LIMIT: 500 lines per file
# SOFT TARGET: 300 lines per file

set -e

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

VIOLATION=0
WARNING=0

echo "=== CLAUDEME File Size Check ==="
echo ""

# Find all Python files and check line counts
while IFS= read -r file; do
    lines=$(wc -l < "$file")
    
    if [ "$lines" -gt 500 ]; then
        echo -e "${RED}VIOLATION${NC}: $file ($lines lines > 500)"
        VIOLATION=$((VIOLATION + 1))
    elif [ "$lines" -gt 300 ]; then
        echo -e "${YELLOW}WARNING${NC}: $file ($lines lines > 300)"
        WARNING=$((WARNING + 1))
    fi
done < <(find ./src ./cli ./tests -name "*.py" -type f 2>/dev/null)

echo ""
echo "=== Summary ==="
echo "Violations (>500 lines): $VIOLATION"
echo "Warnings (>300 lines): $WARNING"

if [ "$VIOLATION" -gt 0 ]; then
    echo -e "${RED}FAILED${NC}: $VIOLATION files exceed 500 line limit"
    exit 1
else
    echo -e "${GREEN}PASSED${NC}: No files exceed 500 line limit"
    exit 0
fi
