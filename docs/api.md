# api reference

Reference for the functions and types exported in this PR. The list grows
as more milestones land.

Layering, top to bottom:

- `cuda` is the user-facing package
- `cudaresult` maps CUresult codes to Go errors
- `cudasys` holds raw types and bound function pointers
- `internal/dynload` opens shared libraries by path
- `internal/platform` returns OS-specific candidate library paths

## package cuda

### func Init() error

Loads the CUDA driver library and calls `cuInit(0)`. Idempotent: the second
and later calls return nil without reloading. The library handle is closed
automatically if `cuInit` fails, so retries do not leak.

```go
if err := cuda.Init(); err != nil {
    log.Fatal(err)
}
```

### func DriverVersion() (int, error)

Returns the installed driver version following the CUDA convention. `12030`
means CUDA 12.3. Returns `cuda.ErrNotInitialized` if `Init` has not run.

```go
v, err := cuda.DriverVersion()
fmt.Printf("driver: %d.%d\n", v/1000, (v%1000)/10)
```

### type Error

Alias for `cudaresult.Error`. Carries a CUresult code and the name of the
driver call that returned it. Compare with `errors.Is` against the sentinel
variables.

```go
if err := cuda.Init(); errors.Is(err, cuda.ErrOperatingSystem) {
    // OS-level call inside cuInit failed
}
```

### sentinels

`ErrInvalidValue`, `ErrOutOfMemory`, `ErrNotInitialized`, `ErrDeinitialized`,
`ErrProfilerDisabled`, `ErrStubLibrary`, `ErrDeviceUnavailable`,
`ErrNoDevice`, `ErrInvalidDevice`, `ErrDeviceNotLicensed`, `ErrInvalidImage`,
`ErrInvalidContext`, `ErrNoBinaryForGPU`, `ErrInvalidPTX`,
`ErrUnsupportedPTXVersion`, `ErrInvalidSource`, `ErrFileNotFound`,
`ErrSharedObjectSymbolNotFound`, `ErrSharedObjectInitFailed`,
`ErrOperatingSystem`, `ErrInvalidHandle`, `ErrIllegalState`, `ErrNotFound`,
`ErrNotReady`, `ErrIllegalAddress`, `ErrLaunchOutOfResources`,
`ErrLaunchTimeout`, `ErrLaunchFailed`, `ErrNotPermitted`, `ErrNotSupported`,
`ErrSystemNotReady`, `ErrSystemDriverMismatch`, `ErrUnknown`.

Returned errors for codes outside the table still match via
`errors.Is(err, &cuda.Error{Code: code})`.

## package cudaresult

### func Init(d *cudasys.Driver, flags uint32) error

Calls `d.CuInit(flags)` and maps the CUresult to a typed error. Returns
`ErrNotInitialized` when `d` or the bound function is nil. Tests at higher
layers construct a fake `*cudasys.Driver` and call this directly.

### func DriverVersion(d *cudasys.Driver) (int, error)

Calls `d.CuDriverGetVersion`. Same nil-driver handling as `Init`.

### type Error

```go
type Error struct {
    Code cudasys.CUresult
    Op   string
}
```

`Error()` renders as `Op: NAME` or just `NAME` when `Op` is empty. Codes
outside the table render as `CUDA_ERROR_<number>`. `Is(target error) bool`
matches on `Code` only, so wrapped errors with extra `Op` context still
match the bare sentinels.

## package cudasys

### type Driver

Holds the bound CUDA driver function pointers and the underlying
`dynload.Library` handle.

```go
type Driver struct {
    CuInit             func(flags uint32) CUresult
    CuDriverGetVersion func(version *int32) CUresult
    // unexported lib
}
```

Tests construct a `Driver` literal with just the function fields set; the
library is left nil and `Close` is a no-op.

### func Load(lib dynload.Library) (*Driver, error)

Binds the v0 symbol set (`cuInit`, `cuDriverGetVersion`). Closes `lib` on
binding failure so callers do not have to track ownership of the handle on
the failure path.

### func (*Driver) Close() error

Releases the underlying library. Safe to call on a nil receiver, on a
Driver without a library, and more than once.

### func (CUresult) Name() string

Returns the upper-case CUDA macro for the result code, or `""` for codes
outside the table. Used by `cudaresult.Error` to render error strings.

### CUresult constants

Numeric values match the public CUDA header. The table covers the codes
documented in CUDA 12.x; additions are mechanical.

## package internal/dynload

### type Library

```go
type Library interface {
    Handle() uintptr
    Close() error
}
```

### type Opener

```go
type Opener interface {
    Open(path string) (Library, error)
}
```

### func OpenAny(o Opener, paths []string) (Library, error)

Tries each path with `o` in order and returns the first success. Joins all
errors when every path fails. Returns `ErrNoCandidates` for an empty list.

### func Default() Opener

OS-specific. Linux returns an Opener backed by `purego.Dlopen`; windows
returns one backed by `syscall.LoadLibrary`; other platforms return an
Opener that always reports `ErrUnsupported`.

## package internal/platform

### func LibraryCandidates() []string

Returns the candidate paths for the CUDA driver library, in trial order.

| OS      | Paths |
|---------|-------|
| linux   | `libcuda.so.1`, `/usr/lib/x86_64-linux-gnu/libcuda.so.1`, `/usr/lib/wsl/lib/libcuda.so.1` |
| windows | `nvcuda.dll` |
| other   | nil |
