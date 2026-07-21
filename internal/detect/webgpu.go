//go:build js

// WebGPU browser backend for the js/wasm reader. It reuses the authoritative
// WGSL kernels verbatim (WebGPU accepts the source string directly) and drives
// them through syscall/js, giving the browser the GPU-resident preprocessing
// the native Vulkan path already has. Every WebGPU step that matters is a
// Promise, and Go's wasm runtime cannot block the goroutine that owns the JS
// event loop; awaitJS parks the calling goroutine on a channel and lets the
// scheduler yield to the event loop until the promise settles. The reader
// drives decode off the JS entry callback, so session methods are free to await.
package detect

import (
	_ "embed"
	"encoding/binary"
	"errors"
	"image"
	"syscall/js"
)

//go:embed shaders/halve_nrgba.wgsl
var webgpuHalveSrc string

var errWebGPUUnavailable = errors.New("jabcode: WebGPU is unavailable")

// awaitJS blocks until the promise settles, yielding to the JS event loop.
func awaitJS(p js.Value) (js.Value, error) {
	type settle struct {
		val js.Value
		err error
	}
	ch := make(chan settle, 1)
	onResolve := js.FuncOf(func(_ js.Value, args []js.Value) any {
		var v js.Value
		if len(args) > 0 {
			v = args[0]
		}
		ch <- settle{val: v}
		return nil
	})
	defer onResolve.Release()
	onReject := js.FuncOf(func(_ js.Value, args []js.Value) any {
		msg := "promise rejected"
		if len(args) > 0 {
			if m := args[0].Get("message"); m.Truthy() {
				msg = m.String()
			} else {
				msg = args[0].String()
			}
		}
		ch <- settle{err: errors.New(msg)}
		return nil
	})
	defer onReject.Release()
	p.Call("then", onResolve).Call("catch", onReject)
	s := <-ch
	return s.val, s.err
}

// webgpuDevice owns the long-lived adapter/device/queue and a per-kernel
// compute-pipeline cache, mirroring the native path's persistent ownership.
type webgpuDevice struct {
	device    js.Value
	queue     js.Value
	pipelines map[string]js.Value

	usageStorage int
	usageCopyDst int
	usageCopySrc int
	usageMapRead int
	mapModeRead  js.Value
}

// webgpuPresent reports whether navigator.gpu exists without assuming a
// navigator global (the Node wasm runner has neither).
func webgpuPresent() bool {
	nav := js.Global().Get("navigator")
	if !nav.Truthy() {
		return false
	}
	return nav.Get("gpu").Truthy()
}

func openWebGPUDevice() (*webgpuDevice, error) {
	if !webgpuPresent() {
		return nil, errWebGPUUnavailable
	}
	gpu := js.Global().Get("navigator").Get("gpu")
	adapter, err := awaitJS(gpu.Call("requestAdapter"))
	if err != nil {
		return nil, err
	}
	if !adapter.Truthy() {
		return nil, errWebGPUUnavailable
	}
	device, err := awaitJS(adapter.Call("requestDevice"))
	if err != nil {
		return nil, err
	}
	if !device.Truthy() {
		return nil, errWebGPUUnavailable
	}
	usage := js.Global().Get("GPUBufferUsage")
	return &webgpuDevice{
		device:       device,
		queue:        device.Get("queue"),
		pipelines:    map[string]js.Value{},
		usageStorage: usage.Get("STORAGE").Int(),
		usageCopyDst: usage.Get("COPY_DST").Int(),
		usageCopySrc: usage.Get("COPY_SRC").Int(),
		usageMapRead: usage.Get("MAP_READ").Int(),
		mapModeRead:  js.Global().Get("GPUMapMode").Get("READ"),
	}, nil
}

func (d *webgpuDevice) close() {
	if d.device.Truthy() {
		d.device.Call("destroy")
	}
}

// pipeline builds and caches one compute pipeline per named WGSL source, so a
// kernel compiles once and is reused across frames.
func (d *webgpuDevice) pipeline(name, src string) js.Value {
	if p, ok := d.pipelines[name]; ok {
		return p
	}
	module := d.device.Call("createShaderModule", map[string]any{"code": src})
	p := d.device.Call("createComputePipeline", map[string]any{
		"layout":  "auto",
		"compute": map[string]any{"module": module, "entryPoint": "main"},
	})
	d.pipelines[name] = p
	return p
}

func (d *webgpuDevice) newBuffer(size, usage int) js.Value {
	return d.device.Call("createBuffer", map[string]any{"size": size, "usage": usage})
}

// writeBytes uploads a Go byte slice into a device buffer without a per-element
// round trip through the JS bridge.
func (d *webgpuDevice) writeBytes(buf js.Value, data []byte) {
	view := js.Global().Get("Uint8Array").New(len(data))
	js.CopyBytesToJS(view, data)
	d.queue.Call("writeBuffer", buf, 0, view)
}

// readBytes maps a COPY_DST|MAP_READ buffer and copies its contents into a Go
// slice. The buffer must already hold the result of a submitted copy.
func (d *webgpuDevice) readBytes(buf js.Value, size int) ([]byte, error) {
	if _, err := awaitJS(buf.Call("mapAsync", d.mapModeRead)); err != nil {
		return nil, err
	}
	view := js.Global().Get("Uint8Array").New(buf.Call("getMappedRange"))
	out := make([]byte, size)
	js.CopyBytesToGo(out, view)
	buf.Call("unmap")
	return out, nil
}

func (d *webgpuDevice) bindGroup(pipeline js.Value, buffers ...js.Value) js.Value {
	entries := make([]any, len(buffers))
	for i, b := range buffers {
		entries[i] = map[string]any{
			"binding":  i,
			"resource": map[string]any{"buffer": b},
		}
	}
	return d.device.Call("createBindGroup", map[string]any{
		"layout":  pipeline.Call("getBindGroupLayout", 0),
		"entries": entries,
	})
}

// packNRGBA returns the image's pixels as a tightly packed r,g,b,a byte run, so
// each pixel reads as a little-endian u32 with red in the low byte - exactly the
// layout halve_nrgba.wgsl expects.
func packNRGBA(in *image.NRGBA) []byte {
	w, h := in.Rect.Dx(), in.Rect.Dy()
	if in.Stride == w*4 && in.Rect.Min.X == 0 && in.Rect.Min.Y == 0 {
		return in.Pix[:w*h*4]
	}
	out := make([]byte, w*h*4)
	base := in.PixOffset(in.Rect.Min.X, in.Rect.Min.Y)
	for y := 0; y < h; y++ {
		copy(out[y*w*4:(y+1)*w*4], in.Pix[base+y*in.Stride:base+y*in.Stride+w*4])
	}
	return out
}

// halveNRGBA box-averages in down to ceil(w/2) x ceil(h/2) on the device,
// reproducing HalveNRGBA bit for bit (the shader and the CPU path share the same
// integer box average). It is the first kernel of the resident pyramid.
func (d *webgpuDevice) halveNRGBA(in *image.NRGBA) (*image.NRGBA, error) {
	w, h := in.Rect.Dx(), in.Rect.Dy()
	nw, nh := max((w+1)/2, 1), max((h+1)/2, 1)
	src := packNRGBA(in)
	dstBytes := nw * nh * 4

	srcBuf := d.newBuffer(len(src), d.usageStorage|d.usageCopyDst)
	d.writeBytes(srcBuf, src)
	dstBuf := d.newBuffer(dstBytes, d.usageStorage|d.usageCopySrc)
	paramsBuf := d.newBuffer(16, d.usageStorage|d.usageCopyDst)
	params := make([]byte, 16)
	binary.LittleEndian.PutUint32(params[0:], uint32(w))
	binary.LittleEndian.PutUint32(params[4:], uint32(h))
	binary.LittleEndian.PutUint32(params[8:], uint32(nw))
	binary.LittleEndian.PutUint32(params[12:], uint32(nh))
	d.writeBytes(paramsBuf, params)

	pipeline := d.pipeline("halve_nrgba", webgpuHalveSrc)
	bind := d.bindGroup(pipeline, srcBuf, dstBuf, paramsBuf)

	enc := d.device.Call("createCommandEncoder")
	pass := enc.Call("beginComputePass")
	pass.Call("setPipeline", pipeline)
	pass.Call("setBindGroup", 0, bind)
	pass.Call("dispatchWorkgroups", (nw+7)/8, (nh+7)/8)
	pass.Call("end")

	readBuf := d.newBuffer(dstBytes, d.usageCopyDst|d.usageMapRead)
	enc.Call("copyBufferToBuffer", dstBuf, 0, readBuf, 0, dstBytes)
	d.queue.Call("submit", []any{enc.Call("finish")})

	outPix, err := d.readBytes(readBuf, dstBytes)
	for _, b := range []js.Value{srcBuf, dstBuf, paramsBuf, readBuf} {
		b.Call("destroy")
	}
	if err != nil {
		return nil, err
	}
	out := image.NewNRGBA(image.Rect(0, 0, nw, nh))
	copy(out.Pix, outPix)
	return out, nil
}
