#!/usr/bin/env bash
# Regenerate vector_add.ptx from vector_add.cu and keep the cuda package's
# test fixture in sync.
#
# Requires the CUDA toolkit (nvcc) on PATH. Not needed to build gocudrv
# itself; only needed to refresh the checked-in PTX after editing the .cu.
set -euo pipefail

cd "$(dirname "$0")"

# Single canonical build command. All other regeneration paths
# (`make ptx`, `go generate ./examples/vector-add`) route through this
# script so the produced PTX is identical regardless of entry point.
nvcc -ptx -arch=sm_50 vector_add.cu -o vector_add.ptx
echo "wrote $(pwd)/vector_add.ptx"

# The cuda package and event-pipeline example reuse the same kernel. Keep all
# copies byte-identical so users can rely on the docs saying they share one
# source artifact.
fixture="../../cuda/testdata/vector_add.ptx"
cp vector_add.ptx "$fixture"
echo "wrote $(realpath "$fixture")"

pipeline="../event-pipeline/vector_add.ptx"
cp vector_add.ptx "$pipeline"
echo "wrote $(realpath "$pipeline")"
