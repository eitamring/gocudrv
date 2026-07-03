package cudasys

// CUipcMemHandle mirrors the driver's opaque 64-byte IPC memory handle
// (CU_IPC_HANDLE_SIZE). It crosses process boundaries as plain bytes.
type CUipcMemHandle struct {
	Data [64]byte
}

// CUipcEventHandle mirrors the driver's opaque 64-byte IPC event handle.
type CUipcEventHandle struct {
	Data [64]byte
}

// IpcMemLazyEnablePeerAccess maps CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS, the only
// flag cuIpcOpenMemHandle accepts.
const IpcMemLazyEnablePeerAccess uint32 = 0x1
