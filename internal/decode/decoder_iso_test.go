package decode

import (
	"bytes"
	"testing"

	"github.com/srlehn/jabcode/internal/wire"
)

type messageBits []byte

func (b *messageBits) write(value, width int) {
	for shift := width - 1; shift >= 0; shift-- {
		*b = append(*b, byte(value>>shift&1))
	}
}

func (b *messageBits) upper(value int) { b.write(value, 5) }

func (b *messageBits) additional(value int) {
	b.write(31, 5)
	b.write(3, 2)
	b.write(value, 3)
}

func (b *messageBits) eci(assignment, width int) {
	b.write(31, 5)
	b.write(2, 2)
	switch width {
	case 8:
		b.write(assignment, width)
	case 16:
		b.write(1<<15|assignment, width)
	case 22:
		b.write(3<<20|assignment, width)
	default:
		panic("unsupported ECI test width")
	}
}

func (b *messageBits) byteRun(values ...byte) {
	if len(values) == 0 || len(values) > 15 {
		panic("unsupported byte-run test length")
	}
	b.write(31, 5)
	b.write(0, 2)
	b.write(len(values), 4)
	for _, value := range values {
		b.write(int(value), 8)
	}
}

func requireISODecode(t *testing.T, bits messageBits, want []byte) {
	t.Helper()
	got, ok := DecodeDataProfile(bits, wire.ISO23634)
	if !ok {
		t.Fatal("DecodeDataProfile rejected a valid ISO message stream")
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("DecodeDataProfile = %q, want %q", got, want)
	}
}

func TestDecodeDataISO23634ModeSwitches(t *testing.T) {
	t.Run("ordinary data advertises ECI-capable reader", func(t *testing.T) {
		var bits messageBits
		bits.upper(1)
		bits.upper(2)
		requireISODecode(t, bits, []byte("]j1AB"))
	})

	t.Run("lowercase numeric shift", func(t *testing.T) {
		var bits messageBits
		bits.upper(28) // latch lowercase
		bits.write(1, 5)
		bits.write(31, 5)
		bits.write(3, 2) // shift numeric for one character
		bits.write(2, 4)
		bits.write(2, 5)
		requireISODecode(t, bits, []byte("]j1a1b"))
	})

	shortcuts := []struct {
		name       string
		additional int
		want       string
	}{
		{"https", 1, "https://"},
		{"http", 2, "http://"},
		{"www", 3, "www."},
	}
	for _, tc := range shortcuts {
		t.Run(tc.name, func(t *testing.T) {
			var bits messageBits
			bits.additional(tc.additional)
			requireISODecode(t, bits, []byte("]j1"+tc.want))
		})
	}

	t.Run("shortcut returns from uppercase shift", func(t *testing.T) {
		var bits messageBits
		bits.upper(28) // latch lowercase
		bits.write(1, 5)
		bits.write(28, 5) // shift uppercase
		bits.additional(1)
		bits.write(2, 5)
		requireISODecode(t, bits, []byte("]j1ahttps://b"))
	})
}

func TestDecodeDataISO23634ECI(t *testing.T) {
	assignments := []struct {
		name       string
		assignment int
		width      int
		transmit   string
	}{
		{"eight bit", 26, 8, "000026"},
		{"sixteen bit", 300, 16, "000300"},
		{"twenty-two bit", 200000, 22, "200000"},
	}
	for _, tc := range assignments {
		t.Run(tc.name, func(t *testing.T) {
			var bits messageBits
			bits.upper(1)
			bits.eci(tc.assignment, tc.width)
			bits.upper(2)
			want := append([]byte("]j1A\\"), tc.transmit...)
			want = append(want, 'B')
			requireISODecode(t, bits, want)
		})
	}

	t.Run("literal backslashes are doubled throughout message", func(t *testing.T) {
		var bits messageBits
		bits.upper(1)
		bits.byteRun('\\')
		bits.upper(2)
		bits.eci(26, 8)
		bits.byteRun('\\')
		bits.upper(3)
		want := []byte{']', 'j', '1', 'A', '\\', '\\', 'B', '\\', '0', '0', '0', '0', '2', '6', '\\', '\\', 'C'}
		requireISODecode(t, bits, want)
	})

	t.Run("literal backslashes are doubled without an assignment", func(t *testing.T) {
		var bits messageBits
		bits.upper(1)
		bits.byteRun('\\')
		bits.upper(2)
		requireISODecode(t, bits, []byte("]j1A\\\\B"))
	})

	t.Run("ECI returns to invoking shifted mode", func(t *testing.T) {
		var bits messageBits
		bits.upper(28) // latch lowercase
		bits.write(1, 5)
		bits.write(28, 5) // shift uppercase
		bits.eci(26, 8)
		bits.upper(1)
		bits.write(2, 5)
		requireISODecode(t, bits, []byte("]j1a\\000026Ab"))
	})
}

func TestDecodeDataISO23634FNC1(t *testing.T) {
	t.Run("before first character", func(t *testing.T) {
		var bits messageBits
		bits.additional(4)
		bits.upper(1)
		bits.additional(4)
		bits.upper(2)
		bits.additional(5)
		want := []byte{']', 'j', '4', 'A', 29, 'B'}
		requireISODecode(t, bits, want)
	})

	t.Run("after initial letter", func(t *testing.T) {
		var bits messageBits
		bits.upper(1)
		bits.additional(4)
		bits.upper(2)
		bits.additional(5)
		requireISODecode(t, bits, []byte("]j5AB"))
	})

	t.Run("after initial digit pair", func(t *testing.T) {
		var bits messageBits
		bits.upper(29) // latch numeric
		bits.write(2, 4)
		bits.write(3, 4)
		bits.write(14, 4) // latch uppercase
		bits.additional(4)
		bits.upper(2)
		bits.additional(5)
		requireISODecode(t, bits, []byte("]j512B"))
	})

	t.Run("ECI and leading FNC1", func(t *testing.T) {
		var bits messageBits
		bits.eci(26, 8)
		bits.additional(4)
		bits.upper(1)
		bits.additional(5)
		requireISODecode(t, bits, []byte("]j4\\000026A"))
	})

	t.Run("ECI and FNC1 after initial letter", func(t *testing.T) {
		var bits messageBits
		bits.upper(1)
		bits.additional(4)
		bits.eci(26, 8)
		bits.upper(2)
		bits.additional(5)
		requireISODecode(t, bits, []byte("]j5A\\000026B"))
	})
}

func TestDecodeDataISO23634ISO15434(t *testing.T) {
	t.Run("ordinary format transmits message trailer", func(t *testing.T) {
		var bits messageBits
		bits.additional(0)
		bits.byteRun('0', '6', 29, '1', 'P', 'A', 'B', 'C', 30)
		bits.additional(5)
		want := []byte{']', 'j', '1', '[', ')', '>', 30, '0', '6', 29, '1', 'P', 'A', 'B', 'C', 30, 4}
		requireISODecode(t, bits, want)
	})

	t.Run("multiple format envelopes", func(t *testing.T) {
		var bits messageBits
		bits.additional(0)
		bits.byteRun('0', '6', 29, '1', 'P', 'A', 'B', 'C', 30, '1', '2', 29, 'B', 30)
		bits.additional(5)
		want := []byte{']', 'j', '1', '[', ')', '>', 30, '0', '6', 29, '1', 'P', 'A', 'B', 'C', 30, '1', '2', 29, 'B', 30, 4}
		requireISODecode(t, bits, want)
	})

	for _, tc := range []struct {
		name string
		data []byte
	}{
		{name: "format 02 omits message trailer", data: []byte("02UNB+1+UNZ")},
		{name: "format 08 omits message trailer", data: []byte("0820250101CII")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var bits messageBits
			bits.additional(0)
			bits.byteRun(tc.data...)
			bits.additional(5)
			want := append([]byte{']', 'j', '1', '[', ')', '>', 30}, tc.data...)
			requireISODecode(t, bits, want)
		})
	}

	t.Run("literal EOT in binary format is data", func(t *testing.T) {
		var bits messageBits
		bits.additional(0)
		bits.byteRun('0', '9', 29, 'B', 'I', 'N', 29, 'N', 'O', 'N', 'E', 29, '1', 29)
		bits.byteRun(4, 30)
		bits.additional(5)
		want := []byte{']', 'j', '1', '[', ')', '>', 30, '0', '9', 29, 'B', 'I', 'N', 29, 'N', 'O', 'N', 'E', 29, '1', 29, 4, 30, 4}
		requireISODecode(t, bits, want)
	})
}

func TestDecodeDataISO23634RejectsInvalidControls(t *testing.T) {
	cases := []struct {
		name string
		bits messageBits
	}{
		{
			name: "truncated ECI",
			bits: func() messageBits {
				var bits messageBits
				bits.write(31, 5)
				bits.write(2, 2)
				bits.write(26, 7)
				return bits
			}(),
		},
		{
			name: "ECI assignment above maximum",
			bits: func() messageBits {
				var bits messageBits
				bits.eci(1000000, 22)
				return bits
			}(),
		},
		{
			name: "reserved additional switch",
			bits: func() messageBits {
				var bits messageBits
				bits.additional(6)
				return bits
			}(),
		},
		{
			name: "unterminated ISO 15434 shift",
			bits: func() messageBits {
				var bits messageBits
				bits.additional(0)
				bits.byteRun('0', '6')
				return bits
			}(),
		},
		{
			name: "truncated ISO 15434 format indicator",
			bits: func() messageBits {
				var bits messageBits
				bits.additional(0)
				bits.byteRun('0')
				bits.additional(5)
				return bits
			}(),
		},
		{
			name: "nonnumeric ISO 15434 format indicator",
			bits: func() messageBits {
				var bits messageBits
				bits.additional(0)
				bits.byteRun('A', '7')
				bits.additional(5)
				return bits
			}(),
		},
		{
			name: "misplaced ISO 15434 shift",
			bits: func() messageBits {
				var bits messageBits
				bits.upper(1)
				bits.additional(0)
				return bits
			}(),
		},
		{
			name: "nested ISO 15434 shift",
			bits: func() messageBits {
				var bits messageBits
				bits.additional(0)
				bits.additional(0)
				return bits
			}(),
		},
		{
			name: "ISO 15434 nested in FNC1",
			bits: func() messageBits {
				var bits messageBits
				bits.additional(4)
				bits.additional(0)
				return bits
			}(),
		},
		{
			name: "FNC1 nested in ISO 15434",
			bits: func() messageBits {
				var bits messageBits
				bits.additional(0)
				bits.additional(4)
				return bits
			}(),
		},
		{
			name: "ISO 15434 after FNC1",
			bits: func() messageBits {
				var bits messageBits
				bits.additional(4)
				bits.additional(5)
				bits.additional(0)
				return bits
			}(),
		},
		{
			name: "FNC1 after ISO 15434",
			bits: func() messageBits {
				var bits messageBits
				bits.additional(0)
				bits.byteRun('0', '2')
				bits.additional(5)
				bits.additional(4)
				return bits
			}(),
		},
		{
			name: "duplicate ISO 15434 terminator",
			bits: func() messageBits {
				var bits messageBits
				bits.additional(0)
				bits.byteRun('0', '6')
				bits.additional(5)
				bits.additional(5)
				return bits
			}(),
		},
		{
			name: "EOT outside structured mode",
			bits: func() messageBits {
				var bits messageBits
				bits.additional(5)
				return bits
			}(),
		},
		{
			name: "FNC1 at invalid position",
			bits: func() messageBits {
				var bits messageBits
				bits.upper(1)
				bits.upper(2)
				bits.additional(4)
				return bits
			}(),
		},
		{
			name: "unterminated FNC1",
			bits: func() messageBits {
				var bits messageBits
				bits.additional(4)
				bits.upper(1)
				return bits
			}(),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := DecodeDataProfile(tc.bits, wire.ISO23634)
			if ok || got != nil {
				t.Fatalf("DecodeDataProfile = (%q, %v), want (nil, false)", got, ok)
			}
		})
	}
}
