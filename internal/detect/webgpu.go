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
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"math"
	"sync"
	"syscall/js"

	"github.com/srlehn/jabcode/internal/core"
)

var errWebGPUUnavailable = errors.New("jabcode: WebGPU is unavailable")

const (
	gpuBinarizerParamsSize = 48
	gpuThresholdCellSize   = 32
)

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
	adapter js.Value
	device  js.Value
	queue   js.Value

	mu        sync.Mutex
	pipelines map[string]js.Value

	usageStorage  int
	usageCopyDst  int
	usageCopySrc  int
	usageMapRead  int
	mapModeRead   js.Value
	maxBufferSize int
	maxWorkgroups int
}

var automaticWebGPURuntime struct {
	mu      sync.Mutex
	adapter js.Value
	device  *webgpuDevice
}

func checkedImageBytes(width, height, channels int) (pixels, bytes int, err error) {
	if width <= 0 || height <= 0 || channels <= 0 {
		return 0, 0, errors.New("webgpu: invalid image dimensions")
	}
	if width > math.MaxInt/height {
		return 0, 0, errors.New("webgpu: image pixel count overflow")
	}
	pixels = width * height
	if channels > math.MaxInt/pixels {
		return 0, 0, errors.New("webgpu: image byte count overflow")
	}
	return pixels, pixels * channels, nil
}

func (d *webgpuDevice) checkBufferSize(size int) error {
	if size <= 0 || (d.maxBufferSize > 0 && size > d.maxBufferSize) {
		return fmt.Errorf("webgpu: buffer size %d exceeds device limits", size)
	}
	return nil
}

func (d *webgpuDevice) checkDispatch(x, y int) error {
	if x <= 0 || y <= 0 || (d.maxWorkgroups > 0 && (x > d.maxWorkgroups || y > d.maxWorkgroups)) {
		return fmt.Errorf("webgpu: dispatch %dx%d exceeds device limits", x, y)
	}
	return nil
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
	automaticWebGPURuntime.mu.Lock()
	defer automaticWebGPURuntime.mu.Unlock()
	if automaticWebGPURuntime.device != nil {
		return automaticWebGPURuntime.device, nil
	}
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
	limits := device.Get("limits")
	maxBufferSize, maxWorkgroups := math.MaxInt, math.MaxInt
	if limits.Truthy() {
		if value := limits.Get("maxBufferSize"); value.Truthy() {
			maxBufferSize = value.Int()
		}
		if value := limits.Get("maxComputeWorkgroupsPerDimension"); value.Truthy() {
			maxWorkgroups = value.Int()
		}
	}
	result := &webgpuDevice{
		adapter:       adapter,
		device:        device,
		queue:         device.Get("queue"),
		pipelines:     map[string]js.Value{},
		usageStorage:  usage.Get("STORAGE").Int(),
		usageCopyDst:  usage.Get("COPY_DST").Int(),
		usageCopySrc:  usage.Get("COPY_SRC").Int(),
		usageMapRead:  usage.Get("MAP_READ").Int(),
		mapModeRead:   js.Global().Get("GPUMapMode").Get("READ"),
		maxBufferSize: maxBufferSize,
		maxWorkgroups: maxWorkgroups,
	}
	automaticWebGPURuntime.adapter = adapter
	automaticWebGPURuntime.device = result
	return result, nil
}

// pipeline builds and caches one compute pipeline per named WGSL source. WGSL
// compilation is expensive, so the cache guarantees each kernel compiles exactly
// once per device; the lock keeps that guarantee if routes ever call it
// concurrently.
func (d *webgpuDevice) pipeline(name, src string) js.Value {
	d.mu.Lock()
	defer d.mu.Unlock()
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

func (d *webgpuDevice) submit(enc js.Value) error {
	d.device.Call("pushErrorScope", "validation")
	d.queue.Call("submit", []any{enc.Call("finish")})
	scope, err := awaitJS(d.device.Call("popErrorScope"))
	if err != nil {
		retireAutomaticWebGPUDevice(d)
		return err
	}
	if scope.Truthy() {
		retireAutomaticWebGPUDevice(d)
		return errors.New("webgpu validation: " + scope.Get("message").String())
	}
	return nil
}

func retireAutomaticWebGPUDevice(device *webgpuDevice) {
	automaticWebGPURuntime.mu.Lock()
	defer automaticWebGPURuntime.mu.Unlock()
	if automaticWebGPURuntime.device == device {
		automaticWebGPURuntime.device = nil
		automaticWebGPURuntime.adapter = js.Value{}
	}
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
	_, srcBytes, err := checkedImageBytes(w, h, 4)
	if err != nil || len(in.Pix) < srcBytes {
		return nil, errWebGPUUnavailable
	}
	src := packNRGBA(in)
	_, dstBytes, err := checkedImageBytes(nw, nh, 4)
	if err != nil || d.checkBufferSize(len(src)) != nil || d.checkBufferSize(dstBytes) != nil {
		return nil, errWebGPUUnavailable
	}
	if err := d.checkDispatch((nw+7)/8, (nh+7)/8); err != nil {
		return nil, err
	}

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

	pipeline := d.pipeline("halve_nrgba", halveNRGBAWGSL)
	bind := d.bindGroup(pipeline, srcBuf, dstBuf, paramsBuf)

	enc := d.device.Call("createCommandEncoder")
	runPass(enc, pipeline, bind, (nw+7)/8, (nh+7)/8)

	readBuf := d.newBuffer(dstBytes, d.usageCopyDst|d.usageMapRead)
	enc.Call("copyBufferToBuffer", dstBuf, 0, readBuf, 0, dstBytes)
	if err := d.submit(enc); err != nil {
		for _, b := range []js.Value{srcBuf, dstBuf, paramsBuf, readBuf} {
			b.Call("destroy")
		}
		return nil, err
	}

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

// balanceRGB stretches channels 0..2 of bm to the full range on the device,
// reproducing BalanceRGB in place. It is the resident chain's first real
// multi-stage stage: a per-pixel histogram, a per-channel min/max reduction and
// the per-pixel stretch, all in one submission. Byte identity holds because the
// integer stretch in balance_rgb.wgsl equals BalanceRGB's float LUT for every
// byte input, and the count-above-20 bound rule is shared.
func (d *webgpuDevice) balanceRGB(bm *core.Bitmap) error {
	w, h, bpp := bm.Width, bm.Height, bm.Channels
	n, total, err := checkedImageBytes(w, h, 4)
	if err != nil || bpp < 3 || len(bm.Pix) < n*bpp {
		return errWebGPUUnavailable
	}
	if err := d.checkBufferSize(total); err != nil {
		return err
	}
	src := make([]byte, total)
	for i := 0; i < n; i++ {
		o := i * bpp
		src[i*4+0] = bm.Pix[o+0]
		src[i*4+1] = bm.Pix[o+1]
		src[i*4+2] = bm.Pix[o+2]
	}

	params := make([]byte, 8)
	binary.LittleEndian.PutUint32(params[0:], uint32(w))
	binary.LittleEndian.PutUint32(params[4:], uint32(h))

	var buffers []js.Value
	defer func() {
		for _, buffer := range buffers {
			buffer.Call("destroy")
		}
	}()
	srcBuf := d.newBuffer(total, d.usageStorage|d.usageCopyDst)
	buffers = append(buffers, srcBuf)
	d.writeBytes(srcBuf, src)
	balancedBuf := d.newBuffer(len(src), d.usageStorage|d.usageCopySrc)
	buffers = append(buffers, balancedBuf)
	histBuf := d.newBuffer(768*4, d.usageStorage|d.usageCopyDst)
	buffers = append(buffers, histBuf)
	boundsBuf := d.newBuffer(6*4, d.usageStorage)
	buffers = append(buffers, boundsBuf)
	paramsBuf := d.newBuffer(8, d.usageStorage|d.usageCopyDst)
	buffers = append(buffers, paramsBuf)
	d.writeBytes(paramsBuf, params)

	histPipeline := d.pipeline("histogram_rgb", histogramRGBWGSL)
	boundsPipeline := d.pipeline("histogram_bounds", histogramBoundsWGSL)
	balancePipeline := d.pipeline("balance_rgb", balanceRGBWGSL)
	histBind := d.bindGroup(histPipeline, srcBuf, histBuf, paramsBuf)
	boundsBind := d.bindGroup(boundsPipeline, histBuf, boundsBuf)
	balanceBind := d.bindGroup(balancePipeline, srcBuf, balancedBuf, boundsBuf, paramsBuf)

	gx, gy := (w+7)/8, (h+7)/8
	if err := d.checkDispatch(gx, gy); err != nil {
		return err
	}
	enc := d.device.Call("createCommandEncoder")
	enc.Call("clearBuffer", histBuf, 0, 768*4)
	runPass(enc, histPipeline, histBind, gx, gy)
	runPass(enc, boundsPipeline, boundsBind, 1, 1)
	runPass(enc, balancePipeline, balanceBind, gx, gy)

	readBuf := d.newBuffer(len(src), d.usageCopyDst|d.usageMapRead)
	buffers = append(buffers, readBuf)
	enc.Call("copyBufferToBuffer", balancedBuf, 0, readBuf, 0, len(src))
	if err := d.submit(enc); err != nil {
		return err
	}

	outPix, err := d.readBytes(readBuf, len(src))
	if err != nil {
		return err
	}
	for i := 0; i < n; i++ {
		o := i * bpp
		bm.Pix[o+0] = outPix[i*4+0]
		bm.Pix[o+1] = outPix[i*4+1]
		bm.Pix[o+2] = outPix[i*4+2]
	}
	return nil
}

// runPass records one compute dispatch into enc.
func runPass(enc, pipeline, bind js.Value, gx, gy int) {
	pass := enc.Call("beginComputePass")
	pass.Call("setPipeline", pipeline)
	pass.Call("setBindGroup", 0, bind)
	pass.Call("dispatchWorkgroups", gx, gy)
	pass.Call("end")
}

type webgpuPyramidLevel struct {
	width, height int
	buffer        js.Value
}

// webgpuPyramid keeps the expensive image chain on the device. Readback is
// deferred until a CPU route actually needs a level, so concurrent routes share
// one upload and one set of half-scale dispatches.
type webgpuPyramid struct {
	device *webgpuDevice
	levels []webgpuPyramidLevel
	params []js.Value
	closed bool
}

func newWebGPUPyramid(device *webgpuDevice, base *image.NRGBA, levelCount int) (*webgpuPyramid, error) {
	if device == nil || base == nil || base.Rect.Empty() || levelCount <= 0 {
		return nil, errWebGPUUnavailable
	}
	pyramid := &webgpuPyramid{device: device, levels: make([]webgpuPyramidLevel, levelCount)}
	for i := range pyramid.levels {
		width, height := base.Rect.Dx(), base.Rect.Dy()
		for j := 0; j < i; j++ {
			width, height = max((width+1)/2, 1), max((height+1)/2, 1)
		}
		pyramid.levels[i] = webgpuPyramidLevel{width: width, height: height}
	}
	_, baseBytesSize, err := checkedImageBytes(base.Rect.Dx(), base.Rect.Dy(), 4)
	if err != nil || len(base.Pix) < baseBytesSize {
		return nil, errWebGPUUnavailable
	}
	baseBytes := packNRGBA(base)
	if err := device.checkBufferSize(len(baseBytes)); err != nil {
		return nil, err
	}
	usage := device.usageStorage | device.usageCopySrc | device.usageCopyDst
	pyramid.levels[0].buffer = device.newBuffer(len(baseBytes), usage)
	device.writeBytes(pyramid.levels[0].buffer, baseBytes)

	pipeline := device.pipeline("halve_nrgba", halveNRGBAWGSL)
	enc := device.device.Call("createCommandEncoder")
	for i := 1; i < levelCount; i++ {
		previous, current := pyramid.levels[i-1], &pyramid.levels[i]
		_, currentBytes, err := checkedImageBytes(current.width, current.height, 4)
		if err != nil {
			pyramid.close()
			return nil, err
		}
		if err := device.checkBufferSize(currentBytes); err != nil {
			pyramid.close()
			return nil, err
		}
		if err := device.checkDispatch((current.width+7)/8, (current.height+7)/8); err != nil {
			pyramid.close()
			return nil, err
		}
		current.buffer = device.newBuffer(currentBytes, usage)
		params := make([]byte, 16)
		binary.LittleEndian.PutUint32(params[0:], uint32(previous.width))
		binary.LittleEndian.PutUint32(params[4:], uint32(previous.height))
		binary.LittleEndian.PutUint32(params[8:], uint32(current.width))
		binary.LittleEndian.PutUint32(params[12:], uint32(current.height))
		paramBuffer := device.newBuffer(len(params), device.usageStorage|device.usageCopyDst)
		device.writeBytes(paramBuffer, params)
		pyramid.params = append(pyramid.params, paramBuffer)
		bind := device.bindGroup(pipeline, previous.buffer, current.buffer, paramBuffer)
		runPass(enc, pipeline, bind, (current.width+7)/8, (current.height+7)/8)
	}
	if err := device.submit(enc); err != nil {
		pyramid.close()
		return nil, err
	}
	return pyramid, nil
}

func (pyramid *webgpuPyramid) download(level int) (*image.NRGBA, error) {
	if pyramid == nil || pyramid.closed || level < 0 || level >= len(pyramid.levels) {
		return nil, errWebGPUUnavailable
	}
	entry := pyramid.levels[level]
	size := entry.width * entry.height * 4
	readBuffer := pyramid.device.newBuffer(size, pyramid.device.usageCopyDst|pyramid.device.usageMapRead)
	enc := pyramid.device.device.Call("createCommandEncoder")
	enc.Call("copyBufferToBuffer", entry.buffer, 0, readBuffer, 0, size)
	if err := pyramid.device.submit(enc); err != nil {
		return nil, err
	}
	data, err := pyramid.device.readBytes(readBuffer, size)
	readBuffer.Call("destroy")
	if err != nil {
		return nil, err
	}
	out := image.NewNRGBA(image.Rect(0, 0, entry.width, entry.height))
	copy(out.Pix, data)
	return out, nil
}

func (pyramid *webgpuPyramid) rotate(level int, crop image.Rectangle, angle float64) (*core.Bitmap, error) {
	if pyramid == nil || pyramid.closed || level < 0 || level >= len(pyramid.levels) {
		return nil, errWebGPUUnavailable
	}
	source := pyramid.levels[level]
	full := image.Rect(0, 0, source.width, source.height)
	if crop.Empty() || crop.Intersect(full) != crop {
		return nil, errWebGPUUnavailable
	}
	rad := angle * math.Pi / 180
	width := max(int(math.Ceil(math.Abs(float64(crop.Dx())*math.Cos(rad))+math.Abs(float64(crop.Dy())*math.Sin(rad)))), 1)
	height := max(int(math.Ceil(math.Abs(float64(crop.Dx())*math.Sin(rad))+math.Abs(float64(crop.Dy())*math.Cos(rad)))), 1)
	_, size, err := checkedImageBytes(width, height, 4)
	if err != nil {
		return nil, err
	}
	if err := pyramid.device.checkBufferSize(size); err != nil {
		return nil, err
	}
	if err := pyramid.device.checkDispatch((width+7)/8, (height+7)/8); err != nil {
		return nil, err
	}
	usage := pyramid.device.usageStorage | pyramid.device.usageCopySrc
	destination := pyramid.device.newBuffer(size, usage)
	params := make([]byte, 48)
	binary.LittleEndian.PutUint32(params[0:], uint32(source.width))
	binary.LittleEndian.PutUint32(params[4:], uint32(source.height))
	binary.LittleEndian.PutUint32(params[8:], uint32(crop.Min.X))
	binary.LittleEndian.PutUint32(params[12:], uint32(crop.Min.Y))
	binary.LittleEndian.PutUint32(params[16:], uint32(crop.Dx()))
	binary.LittleEndian.PutUint32(params[20:], uint32(crop.Dy()))
	binary.LittleEndian.PutUint32(params[24:], uint32(width))
	binary.LittleEndian.PutUint32(params[28:], uint32(height))
	binary.LittleEndian.PutUint32(params[32:], math.Float32bits(float32(math.Cos(rad))))
	binary.LittleEndian.PutUint32(params[36:], math.Float32bits(float32(math.Sin(rad))))
	paramBuffer := pyramid.device.newBuffer(len(params), pyramid.device.usageStorage|pyramid.device.usageCopyDst)
	pyramid.device.writeBytes(paramBuffer, params)
	defer destination.Call("destroy")
	defer paramBuffer.Call("destroy")
	pipeline := pyramid.device.pipeline("rotate_nrgba", rotateNRGBAWGSL)
	bind := pyramid.device.bindGroup(pipeline, source.buffer, destination, paramBuffer)
	enc := pyramid.device.device.Call("createCommandEncoder")
	runPass(enc, pipeline, bind, (width+7)/8, (height+7)/8)
	readBuffer := pyramid.device.newBuffer(size, pyramid.device.usageCopyDst|pyramid.device.usageMapRead)
	defer readBuffer.Call("destroy")
	enc.Call("copyBufferToBuffer", destination, 0, readBuffer, 0, size)
	if err := pyramid.device.submit(enc); err != nil {
		return nil, err
	}
	data, err := pyramid.device.readBytes(readBuffer, size)
	if err != nil {
		return nil, err
	}
	return &core.Bitmap{Width: width, Height: height, Channels: 4, Pix: data}, nil
}

func (pyramid *webgpuPyramid) close() {
	if pyramid == nil || pyramid.closed {
		return
	}
	pyramid.closed = true
	for _, level := range pyramid.levels {
		if level.buffer.Truthy() {
			level.buffer.Call("destroy")
		}
	}
	for _, params := range pyramid.params {
		params.Call("destroy")
	}
}

// webgpuBinarizeRGB runs the scale-adaptive threshold, RGB classifier, binary
// majority filter and mask packer as one ordered WebGPU submission. The packed
// result crosses back once; detector integration remains separate until this
// complete chain has parity coverage.
func (d *webgpuDevice) webgpuBinarizeRGB(bm *core.Bitmap, printLevels bool) ([3]*core.Bitmap, error) {
	var empty [3]*core.Bitmap
	if d == nil || bm == nil || bm.Width <= 0 || bm.Height <= 0 || bm.Channels != 4 {
		return empty, errWebGPUUnavailable
	}
	pixelCount, imageBytes, err := checkedImageBytes(bm.Width, bm.Height, 4)
	if err != nil || len(bm.Pix) != imageBytes {
		return empty, errWebGPUUnavailable
	}
	if err := d.checkBufferSize(imageBytes); err != nil {
		return empty, err
	}
	blockSize := capInt(min(bm.Width, bm.Height)/binThresholdDivisor, binMinBlock, binMaxBlock)
	blocksX := (bm.Width + blockSize - 1) / blockSize
	blocksY := (bm.Height + blockSize - 1) / blockSize
	_, thresholdBytes, err := checkedImageBytes(blocksX, blocksY, gpuThresholdCellSize)
	if err != nil || pixelCount > (math.MaxInt-7)/4 {
		return empty, errWebGPUUnavailable
	}
	packedSize := ((pixelCount + 7) / 8) * 4
	if err := d.checkBufferSize(thresholdBytes); err != nil {
		return empty, err
	}
	if err := d.checkBufferSize(pixelCount * 4); err != nil {
		return empty, err
	}
	if err := d.checkBufferSize(packedSize); err != nil {
		return empty, err
	}
	params := make([]byte, gpuBinarizerParamsSize)
	binary.LittleEndian.PutUint32(params[0:], uint32(bm.Width))
	binary.LittleEndian.PutUint32(params[4:], uint32(bm.Height))
	binary.LittleEndian.PutUint32(params[8:], uint32(blockSize))
	binary.LittleEndian.PutUint32(params[12:], uint32(blocksX))
	binary.LittleEndian.PutUint32(params[16:], uint32(blocksY))
	if printLevels {
		binary.LittleEndian.PutUint32(params[20:], 2)
	}

	usage := d.usageStorage | d.usageCopyDst | d.usageCopySrc
	input := d.newBuffer(len(bm.Pix), usage)
	d.writeBytes(input, bm.Pix)
	thresholds := d.newBuffer(thresholdBytes, usage)
	rawMasks := d.newBuffer(pixelCount*4, usage)
	finalMasks := d.newBuffer(pixelCount*4, usage)
	packed := d.newBuffer(packedSize, usage)
	paramsBuffer := d.newBuffer(len(params), d.usageStorage|d.usageCopyDst)
	d.writeBytes(paramsBuffer, params)
	defer func() {
		for _, buffer := range []js.Value{input, thresholds, rawMasks, finalMasks, packed, paramsBuffer} {
			buffer.Call("destroy")
		}
	}()

	thresholdPipeline := d.pipeline("block_thresholds", blockThresholdsWGSL)
	classifyPipeline := d.pipeline("binarize_rgb", binarizeRGBWGSL)
	filterPipeline := d.pipeline("filter_binary", filterBinaryWGSL)
	packPipeline := d.pipeline("pack_binary_masks", packBinaryMasksWGSL)
	thresholdBind := d.bindGroup(thresholdPipeline, input, thresholds, paramsBuffer)
	classifyBind := d.bindGroup(classifyPipeline, input, thresholds, rawMasks, paramsBuffer)
	filterBind := d.bindGroup(filterPipeline, rawMasks, finalMasks, paramsBuffer)
	packBind := d.bindGroup(packPipeline, finalMasks, packed, paramsBuffer)
	enc := d.device.Call("createCommandEncoder")
	if err := d.checkDispatch(blocksX, blocksY); err != nil {
		return empty, err
	}
	if err := d.checkDispatch((bm.Width+7)/8, (bm.Height+7)/8); err != nil {
		return empty, err
	}
	if err := d.checkDispatch((packedSize/4+63)/64, 1); err != nil {
		return empty, err
	}
	runPass(enc, thresholdPipeline, thresholdBind, blocksX, blocksY)
	runPass(enc, classifyPipeline, classifyBind, (bm.Width+7)/8, (bm.Height+7)/8)
	runPass(enc, filterPipeline, filterBind, (bm.Width+7)/8, (bm.Height+7)/8)
	runPass(enc, packPipeline, packBind, (packedSize/4+63)/64, 1)
	readBuffer := d.newBuffer(packedSize, d.usageCopyDst|d.usageMapRead)
	defer readBuffer.Call("destroy")
	enc.Call("copyBufferToBuffer", packed, 0, readBuffer, 0, packedSize)
	if err := d.submit(enc); err != nil {
		return empty, err
	}
	packedBytes, err := d.readBytes(readBuffer, packedSize)
	if err != nil {
		return empty, err
	}
	return unpackWebGPUPackedMasks(bm, packedBytes), nil
}

func unpackWebGPUPackedMasks(bm *core.Bitmap, packed []byte) [3]*core.Bitmap {
	var result [3]*core.Bitmap
	for channel := range result {
		result[channel] = core.NewBitmap(bm.Width, bm.Height, 1)
	}
	pixelCount := bm.Width * bm.Height
	for word := 0; word < (pixelCount+7)/8; word++ {
		value := binary.LittleEndian.Uint32(packed[word*4:])
		for lane := 0; lane < 8 && word*8+lane < pixelCount; lane++ {
			mask := value & 7
			pixel := word*8 + lane
			result[0].Pix[pixel] = boolByte(mask&1 != 0)
			result[1].Pix[pixel] = boolByte(mask&2 != 0)
			result[2].Pix[pixel] = boolByte(mask&4 != 0)
			value >>= 3
		}
	}
	return result
}

func boolByte(value bool) byte {
	if value {
		return 255
	}
	return 0
}
