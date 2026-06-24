package cuda

import (
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

func TestRawAccessors(t *testing.T) {
	drv := &cudasys.Driver{}
	c := &Context{raw: 0xC0FFEE, driver: drv}
	if got := c.Raw(); got != 0xC0FFEE {
		t.Errorf("Context.Raw() = %#x, want 0xc0ffee", got)
	}
	if got := c.Driver(); got != drv {
		t.Errorf("Context.Driver() = %p, want %p", got, drv)
	}

	s := &Stream{raw: 0x5151}
	if got := s.Raw(); got != 0x5151 {
		t.Errorf("Stream.Raw() = %#x, want 0x5151", got)
	}

	e := &Event{raw: 0xE7E7}
	if got := e.Raw(); got != 0xE7E7 {
		t.Errorf("Event.Raw() = %#x, want 0xe7e7", got)
	}

	b := &Buffer[float32]{ptr: 0xDEAD}
	if got := b.DevicePtr(); got != 0xDEAD {
		t.Errorf("Buffer.DevicePtr() = %#x, want 0xdead", got)
	}
}

func TestRawAccessorsNilReceiver(t *testing.T) {
	var c *Context
	if c.Raw() != 0 {
		t.Error("nil Context.Raw() = nonzero, want 0")
	}
	if c.Driver() != nil {
		t.Error("nil Context.Driver() = non-nil, want nil")
	}
	var s *Stream
	if s.Raw() != 0 {
		t.Error("nil Stream.Raw() = nonzero, want 0")
	}
	var e *Event
	if e.Raw() != 0 {
		t.Error("nil Event.Raw() = nonzero, want 0")
	}
	var b *Buffer[float32]
	if b.DevicePtr() != 0 {
		t.Error("nil Buffer.DevicePtr() = nonzero, want 0")
	}
}
