BINARY := jabcode
CMD    := ./cmd/jabcode
TAGS   := jabcode_high_color,jabcode_bsi,jabcode_legacy,jabcode_non_iso_encode,osusergo,netgo,static_build

.PHONY: all
all: build

.PHONY: build
build:
	@env CGO_ENABLED=0 go build \
		-tags "$(TAGS)" \
		-trimpath \
		-ldflags '-s -w -extldflags "-fno-PIC -static"' \
		-buildmode pie \
		-o "$(BINARY)" "$(CMD)"

.PHONY: clean
clean:
	@rm -f -- "$(BINARY)"
