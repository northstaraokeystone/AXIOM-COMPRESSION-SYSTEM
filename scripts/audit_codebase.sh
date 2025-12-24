#!/bin/bash
# CLAUDEME Guardrail: Codebase Audit Script

echo "=== SpaceProof Codebase Audit ==="
echo "Date: $(date -Iseconds)"
echo ""

echo "=== File Counts by Directory ==="
for dir in src cli tests benchmarks; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -name "*.py" -type f 2>/dev/null | wc -l)
        echo "$dir: $count files"
    fi
done
echo ""

echo "=== Top 20 Largest Files ==="
find . -name "*.py" -type f -exec wc -l {} \; 2>/dev/null | sort -rn | head -20
echo ""

echo "=== Files > 500 Lines (VIOLATIONS) ==="
find . -name "*.py" -type f -exec wc -l {} \; 2>/dev/null | awk '$1 > 500 {print}'
echo ""

echo "=== Files 300-500 Lines (WARNINGS) ==="
find . -name "*.py" -type f -exec wc -l {} \; 2>/dev/null | awk '$1 > 300 && $1 <= 500 {print}'
echo ""

echo "=== Missing __init__.py ==="
for dir in $(find ./src ./cli -type d 2>/dev/null); do
    if [ -n "$(find "$dir" -maxdepth 1 -name "*.py" -type f 2>/dev/null)" ]; then
        if [ ! -f "$dir/__init__.py" ]; then
            echo "Missing: $dir/__init__.py"
        fi
    fi
done
echo ""

echo "=== Total Line Count ==="
total=$(find . -name "*.py" -type f -exec cat {} \; 2>/dev/null | wc -l)
echo "Total lines: $total"
