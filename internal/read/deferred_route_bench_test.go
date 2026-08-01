//go:build !js

package read

import (
	"bytes"
	"image"
	"os"
	"testing"
	"time"

	_ "image/jpeg"
	_ "image/png"

	_ "golang.org/x/image/webp"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/encode"
)

// benchImageEnv substitutes one image file for the synthetic symbol. The
// device-replay tiers cost one dispatch per candidate finder hit, and a clean
// synthetic symbol has barely more hits than it has finders, so it measures the
// floor of the difference rather than its working size. An adverse capture,
// where the scan produces many hits that the cross-check then rejects, is the
// case the route policy was actually chosen on.
const benchImageEnv = "JABCODE_BENCH_IMAGE"

// benchReverseEnv swaps which arm is timed first, so a reported difference can
// be checked against its own ordering.
const benchReverseEnv = "JABCODE_BENCH_REVERSE"

// benchArmImage is the frame both arms read: a synthetic symbol by default so
// the benchmark runs from a clean checkout, or whatever $JABCODE_BENCH_IMAGE
// names. Captures stay outside the repository, so the path is supplied rather
// than referenced.
// It also returns the payload both arms must produce, which is known only for
// the synthetic symbol; a supplied capture carries no ground truth here, so the
// arms are then held to each other rather than to an absolute.
func benchArmImage(b *testing.B) (image.Image, []byte) {
	b.Helper()
	path := os.Getenv(benchImageEnv)
	if path == "" {
		payload := []byte("deferred mask route arm benchmark")
		img, err := encode.Run(encode.Config{
			Colors: 8, ModuleSize: 12, SymbolNumber: 1,
		}, payload)
		if err != nil {
			b.Fatalf("encode arm benchmark symbol: %v", err)
		}
		return img, isoPayload(payload)
	}
	f, err := os.Open(path)
	if err != nil {
		b.Fatalf("open %s: %v", benchImageEnv, err)
	}
	defer func() { _ = f.Close() }()
	img, _, err := image.Decode(f)
	if err != nil {
		b.Fatalf("decode %s: %v", benchImageEnv, err)
	}
	return img, nil
}

// BenchmarkGPUDecodeRouteArms times the two route policies against each other:
// scan-only, where the device seeds finder row hits and the bit-identical CPU
// chain classifies them, against device replay, where the per-hit cross-check
// chains and the resident pitch fold run on the device and the masks can stay
// packed. Replay is now the default for every context; scan-only remains as the
// twin-exercising seam and as the mode any context runs in until its kernels
// compile. The choice rests on a measurement, so the measurement has to be
// repeatable.
//
// Three things here are what make the comparison mean anything, and all three
// were got wrong by an earlier attempt at this from the outside:
//
//   - Both arms run in one process on one device against one prepared image.
//     Comparing two CLI invocations instead measures process startup, driver
//     pipeline-cache state and image preparation, none of which this fork
//     changes.
//   - Both arms block on compilation of every kernel the policy switches -
//     finder chains and pitch lag both - before they are timed. Replay only
//     engages once those exist, so an unwaited arm quietly measures the CPU twin
//     instead, and waiting on only one of the two lets the other change
//     mechanism mid-run. Waiting is not the same as sleeping first: the driver's
//     cache is warm or cold depending on history, and a sleep tuned for one is
//     wrong for the other.
//   - Each arm must independently produce the expected payload before any
//     duration is reported. Only the bytes are compared: stage, finding,
//     candidate set, arithmetic and search order are all free to differ,
//     because a device tier is allowed to reach the same payload its own way.
//     What a timing ratio cannot survive is the arms having done different
//     work - comparing a successful read against a failed one is not a
//     speedup - so a payload mismatch stops the comparison rather than
//     constraining either implementation.
//
// What it still does not settle: Go runs each sub-benchmark's repeats
// consecutively, so slow thermal or clock drift over the whole run remains
// confounded with arm order even though the cold-start drift is warmed away.
// $JABCODE_BENCH_REVERSE swaps the arms so a result can be checked against its
// own ordering; a difference that survives both orders is the arms, one that
// flips with them is drift.
//
// The default synthetic symbol measures the floor of the difference; see
// benchImageEnv for the case the policy was chosen on. Every figure here is a
// single-level locate, not a whole Decode call, and so is not a wall claim for
// the CLI.
func BenchmarkGPUDecodeRouteArms(b *testing.B) {
	img, expected := benchArmImage(b)
	base := core.BitmapFromImage(img)

	device, err := vulki.Open()
	if err != nil {
		b.Skipf("Vulkan unavailable: %v", err)
	}
	b.Cleanup(func() { _ = device.Close() })
	b.Logf("Vulkan adapter: %s, image %dx%d", device.Info().AdapterName, base.Width, base.Height)

	arms := []struct {
		name string
		open func(*vulki.Device, *core.Bitmap, int) (*detect.GPUDecodeSession, error)
	}{
		{"scan-only", detect.NewGPUDecodeSessionWithDeviceScanOnly},
		{"device-replay", detect.NewGPUDecodeSessionWithDevice},
	}
	if os.Getenv(benchReverseEnv) != "" {
		arms[0], arms[1] = arms[1], arms[0]
	}

	read := func(session *detect.GPUDecodeSession) (*Message, readStage, bool, finding) {
		var f finding
		data, stage, evidence := decodePyramidLevelFindingCapabilities(
			func() image.Image { return img },
			nil, &f, nil, compiledCapabilities(), session, 0,
		)
		return data, stage, evidence, f
	}

	type outcome struct {
		data     []byte
		stage    readStage
		evidence bool
		finding  finding
	}
	sessions := make([]*detect.GPUDecodeSession, len(arms))
	results := make([]outcome, len(arms))
	for i, arm := range arms {
		session, err := arm.open(device, base, 1)
		if err != nil || session == nil {
			b.Skipf("%s session unavailable: %v", arm.name, err)
		}
		b.Cleanup(func() { _ = session.Close() })
		if err := session.WaitReplayKernels(); err != nil {
			b.Skipf("%s finder chains did not compile: %v", arm.name, err)
		}
		sessions[i] = session
		data, stage, evidence, f := read(session)
		results[i] = outcome{messageTransmission(data), stage, evidence, f}
	}
	// Payload only. The arms may reach it by different stages, findings and
	// candidate sets, and a device tier is free to diverge from its CPU twin;
	// but a ratio between one arm that read the symbol and another that did
	// not is not a speedup, so differing bytes stop the comparison.
	if !bytes.Equal(results[0].data, results[1].data) {
		b.Fatalf(
			"route arms produced different payloads, so their timings are not comparable:\n %s: %q\n %s: %q",
			arms[0].name, results[0].data, arms[1].name, results[1].data,
		)
	}
	if want := expected; want != nil && !bytes.Equal(results[0].data, want) {
		b.Fatalf("arms decoded %q, want the encoded payload %q", results[0].data, want)
	}
	b.Logf(
		"both arms: bytes=%d stages=%v/%v",
		len(results[0].data), results[0].stage, results[1].stage,
	)

	// Go runs a sub-benchmark's repeats consecutively, so the first arm
	// absorbs whatever the process is still settling - allocator growth, clock
	// and cache state, the driver's first submissions - and hands the second a
	// warmer machine. Measured cold, that drift was the same size as the gap
	// between the arms, and it reversed which arm won. Both arms therefore run
	// untimed and alternately until it flattens. The budget is wall time and
	// not a pass count because one pass is milliseconds on a synthetic symbol
	// and a quarter second on a full-resolution capture.
	warmup := time.Now().Add(2 * time.Second)
	for pass := 0; pass < 8 || time.Now().Before(warmup); pass++ {
		for _, session := range sessions {
			read(session)
		}
	}

	for i, arm := range arms {
		b.Run(arm.name, func(b *testing.B) {
			for b.Loop() {
				read(sessions[i])
			}
		})
	}
}
