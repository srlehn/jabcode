//go:build !js

package jabcode

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"
)

var wasmMetricPattern = regexp.MustCompile(
	`WASM_METRIC first_decode_ns=([0-9]+) locked_replay_ns=([0-9]+) no_symbol_ns=([0-9]+) capacity=([0-9]+) image_width=([0-9]+) image_height=([0-9]+)`,
)

func TestWasmOpaqueExecutionGate(t *testing.T) {
	temp := t.TempDir()
	binary := filepath.Join(temp, "jabcode.wasm")
	env := append(os.Environ(), "CGO_ENABLED=0", "GOOS=js", "GOARCH=wasm")

	build := exec.Command("go", "build", "-o", binary, "./internal/wasmgate")
	build.Env = env
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build wasm execution test: %v\n%s", err, output)
	}
	info, err := os.Stat(binary)
	if err != nil {
		t.Fatalf("stat wasm execution test: %v", err)
	}

	goRootCommand := exec.Command("go", "env", "GOROOT")
	goRootOutput, err := goRootCommand.Output()
	if err != nil {
		t.Fatalf("go env GOROOT: %v", err)
	}
	executor := filepath.Join(strings.TrimSpace(string(goRootOutput)), "lib", "wasm", "go_js_wasm_exec")
	if _, err := os.Stat(executor); err != nil {
		t.Fatalf("regular-Go wasm executor: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	command := exec.CommandContext(ctx, executor, binary)
	stdout, err := command.StdoutPipe()
	if err != nil {
		t.Fatalf("wasm stdout: %v", err)
	}
	var stderr bytes.Buffer
	command.Stderr = &stderr
	started := time.Now()
	if err := command.Start(); err != nil {
		t.Fatalf("start wasm execution test: %v", err)
	}

	ready := time.Duration(-1)
	var output strings.Builder
	scanner := bufio.NewScanner(stdout)
	for scanner.Scan() {
		line := scanner.Text()
		fmt.Fprintln(&output, line)
		if line == "WASM_READY" && ready < 0 {
			ready = time.Since(started)
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("read wasm execution output: %v", err)
	}
	if err := command.Wait(); err != nil {
		t.Fatalf("run wasm execution test: %v\n%s%s", err, output.String(), stderr.String())
	}
	if ready < 0 {
		t.Fatalf("wasm execution test did not report readiness:\n%s", output.String())
	}
	match := wasmMetricPattern.FindStringSubmatch(output.String())
	if match == nil {
		t.Fatalf("wasm execution test did not report metrics:\n%s", output.String())
	}
	metric := func(index int) time.Duration {
		value, err := strconv.ParseInt(match[index], 10, 64)
		if err != nil {
			t.Fatalf("parse wasm metric %q: %v", match[index], err)
		}
		return time.Duration(value)
	}
	t.Logf(
		"regular-Go wasm: artifact=%d bytes startup=%s first=%s locked=%s no-symbol=%s capacity=%s geometry=%sx%s",
		info.Size(), ready.Round(time.Millisecond), metric(1).Round(time.Millisecond),
		metric(2).Round(time.Millisecond), metric(3).Round(time.Millisecond), match[4], match[5], match[6],
	)
}
