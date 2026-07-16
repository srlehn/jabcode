BINARY := jabcode
CMD    := ./cmd/jabcode
TAGS   := jabcode_high_color,jabcode_bsi,jabcode_legacy,jabcode_non_iso_encode

# The binary stays dynamically linked: the GPU path loads libvulkan at
# runtime through purego's dlopen, which needs the system dynamic loader.
.PHONY: all
all: build

.PHONY: build
build:
	@env CGO_ENABLED=0 go build \
		-tags "$(TAGS)" \
		-trimpath \
		-ldflags '-s -w' \
		-buildmode pie \
		-o "$(BINARY)" "$(CMD)"

.PHONY: clean
clean:
	@rm -f -- "$(BINARY)"
