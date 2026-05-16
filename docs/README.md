# docs

This directory holds focused notes for the current API surface and project
layout. The package is still early, so these pages describe what exists now
rather than the full roadmap.

## pages

- [Getting started](getting-started.md): build, test, WSL2 notes, and the
  `device-info` and `vector-add` examples.
- [Writing and shipping kernels](kernels.md): the `.cu` to `.ptx` to
  `LoadModule` workflow, with both `//go:embed` and load-from-disk
  patterns.
- [Public API](api/cuda.md): exported `cuda` package functions, types, and
  error behavior.
- [Internals](internals.md): package layering, dynamic loading, raw bindings,
  and CUDA result mapping.

## package layout

```text
cuda/          public Go API
cudaresult/    CUresult-to-error helpers
cudasys/       raw CUDA Driver API types and bound functions
internal/      dynamic loader, platform paths, executor, arg packing
examples/      runnable examples
scripts/       build and check helpers
```
