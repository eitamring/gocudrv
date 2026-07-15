# docs

This directory holds focused notes for the current API surface and project
layout.

## pages

- [Getting started](getting-started.md): build, test, WSL2 notes, and the
  `device-info`, `vector-add`, and `event-pipeline` examples.
- [Writing and shipping kernels](kernels.md): the `.cu` to `.ptx` to
  `LoadModule` workflow, with both `//go:embed` and load-from-disk
  patterns.
- [Public API](api/cuda.md): exported `cuda` package functions, types, and
  error behavior.
- [Sibling module integration](integration.md): the contract for modules built
  on top of gocudrv (handle ownership, context and stream sharing, buffer
  lifetime, optional symbols).
- [Internals](internals.md): package layering, dynamic loading, raw bindings,
  and CUDA result mapping.

## package layout

```text
cuda/          public API, tests, and benchmarks
cudaresult/    CUDA result-to-error wrappers
cudasys/       raw Driver API types and dynamic symbols
internal/      loader, executor, arg packing, platform paths, and host callbacks
docs/          guides, API reference, and internals
examples/      runnable examples
scripts/       build and check helpers
```
