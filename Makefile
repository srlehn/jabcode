BINARY := jabcode
CMD    := ./cmd/jabcode
TAGS   := jabcode_high_color,jabcode_bsi,jabcode_legacy,jabcode_non_iso_encode
GO_TOOL := $(if $(GO),$(GO),$(shell command -v go1.27rc2 2>/dev/null || printf go))
GOEXPERIMENT ?= simd

# The binary stays dynamically linked: the GPU path loads libvulkan at
# runtime through purego's dlopen, which needs the system dynamic loader.
.PHONY: all
all: build

.PHONY: build
build:
	@env CGO_ENABLED=0 GOEXPERIMENT="$(GOEXPERIMENT)" "$(GO_TOOL)" build \
		-tags "$(TAGS)" \
		-trimpath \
		-ldflags '-s -w' \
		-buildmode pie \
		-o "$(BINARY)" "$(CMD)"

.PHONY: clean
clean:
	@rm -f -- "$(BINARY)"
