//go:build js

package detect

import (
	"errors"
	"testing"
)

func TestWebGPULimitChecks(t *testing.T) {
	if pixels, bytes, err := checkedImageBytes(17, 9, 4); err != nil || pixels != 153 || bytes != 612 {
		t.Fatalf("checked image size = (%d, %d, %v)", pixels, bytes, err)
	}
	if _, _, err := checkedImageBytes(0, 9, 4); err == nil {
		t.Fatal("accepted empty image dimensions")
	}
	device := &webgpuDevice{maxBufferSize: 1024, maxWorkgroups: 64}
	if err := device.checkBufferSize(1024); err != nil {
		t.Fatalf("accepted buffer limit: %v", err)
	}
	if err := device.checkBufferSize(1025); err == nil {
		t.Fatal("accepted oversized buffer")
	}
	if err := device.checkDispatch(64, 64); err != nil {
		t.Fatalf("accepted dispatch limit: %v", err)
	}
	if err := device.checkDispatch(65, 1); err == nil {
		t.Fatal("accepted oversized dispatch")
	}
}

func TestWebGPUDecodeSessionFailureRetiresResources(t *testing.T) {
	device := &webgpuDevice{}
	session := &GPUDecodeSession{
		device:  device,
		pyramid: &webgpuPyramid{device: device},
	}
	if got := session.failLocked(errors.New("device lost")); got == nil {
		t.Fatal("failure was not returned")
	}
	if !session.closed || session.device != nil || session.pyramid != nil {
		t.Fatal("failed session remained usable")
	}
}
