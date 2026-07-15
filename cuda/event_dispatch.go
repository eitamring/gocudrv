package cuda

import (
	"context"
	"sync"

	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
)

type eventElapsedOp struct {
	driver     *cudasys.Driver
	start, end cudasys.CUevent
	ms         float32
}

func (o *eventElapsedOp) Run() error {
	ms, err := cudaresult.EventElapsedTime(o.driver, o.start, o.end)
	o.ms = ms
	return err
}

var eventElapsedOpPool = sync.Pool{New: func() any { return new(eventElapsedOp) }}

// recycle stays on the caller side because eventElapsed reads ms after doJob.
// Do not add an executor Recycle method to eventElapsedOp.
func (o *eventElapsedOp) reset() {
	*o = eventElapsedOp{}
}

func (o *eventElapsedOp) recycle() {
	o.reset()
	eventElapsedOpPool.Put(o)
}

func (c *Context) eventElapsed(start, end cudasys.CUevent) (float32, error) {
	if c == nil {
		return 0, ErrNilContext
	}
	o := eventElapsedOpPool.Get().(*eventElapsedOp)
	o.driver, o.start, o.end = c.driver, start, end
	err := c.doJob(context.Background(), o)
	ms := o.ms
	o.recycle()
	if err != nil {
		return 0, err
	}
	return ms, nil
}
