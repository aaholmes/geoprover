#!/bin/bash
# Fast test suite (<1 min after initial compile)
# Runs all unit tests, non-ignored integration tests, and Python tests.
set -e

echo "=== Rust tests (unit + fast integration) ==="
cargo test

echo ""
echo "=== Python NN tests (fast only) ==="
python python/test_nn.py

echo ""
echo "=== Python bridge tests ==="
python python/test_bridge.py

echo ""
echo "All fast tests passed."
