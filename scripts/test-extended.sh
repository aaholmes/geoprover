#!/bin/bash
# Extended test suite (includes slow benchmarks + clippy)
# Runs everything: unit tests, all integration tests (including ignored), Python tests, and clippy.
set -e

echo "=== Rust tests (all, including ignored benchmarks) ==="
cargo test -- --include-ignored

echo ""
echo "=== Python NN tests (including slow) ==="
python python/test_nn.py --include-slow

echo ""
echo "=== Python bridge tests ==="
python python/test_bridge.py

echo ""
echo "=== Clippy ==="
cargo clippy -- -D warnings

echo ""
echo "All extended tests passed."
