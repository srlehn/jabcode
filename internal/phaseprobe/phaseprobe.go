// Package phaseprobe provides opt-in process-timeline instrumentation for GPU
// route diagnostics. It is inert unless the caller explicitly enables it.
package phaseprobe

import (
	"fmt"
	"io"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

const eventCapacity = 256

type event struct {
	at     time.Time
	label  string
	detail string
}

type sink struct {
	started time.Time
	next    atomic.Uint32
	events  [eventCapacity]atomic.Pointer[event]

	// Counters accumulate repeated measurements a per-event stamp could not
	// hold: a decode crosses the host-device boundary hundreds of times, and
	// what matters is the total volume per reason, not each crossing.
	countMu  sync.Mutex
	counters map[string]*counter
}

type counter struct {
	ops   int64
	bytes int64
}

var active atomic.Pointer[sink]

// Enable starts a new timing collection for this process.
func Enable() {
	active.Store(&sink{started: time.Now()})
}

// Enabled reports whether coarse phase timing was requested.
func Enabled() bool { return active.Load() != nil }

// Mark writes one absolute and process-relative timestamp. Detail must remain
// tab-free so analyzers can parse the stream without escaping.
func Mark(label string) {
	mark(label, "")
}

// Markf formats event detail only while collection is enabled.
func Markf(label, format string, args ...any) {
	if active.Load() == nil {
		return
	}
	mark(label, fmt.Sprintf(format, args...))
}

func mark(label, detail string) {
	s := active.Load()
	if s == nil {
		return
	}
	index := s.next.Add(1) - 1
	if index >= eventCapacity {
		return
	}
	s.events[index].Store(&event{at: time.Now(), label: label, detail: detail})
}

// Count records one labelled transfer of size bytes. It is a no-op unless
// collection is enabled, so an ordinary decode pays one atomic load per
// crossing and never takes the lock.
func Count(label string, bytes int) {
	s := active.Load()
	if s == nil {
		return
	}
	s.countMu.Lock()
	defer s.countMu.Unlock()
	if s.counters == nil {
		s.counters = make(map[string]*counter)
	}
	c := s.counters[label]
	if c == nil {
		c = &counter{}
		s.counters[label] = c
	}
	c.ops++
	c.bytes += int64(bytes)
}

// Dump writes a stable timestamp-ordered snapshot. Event publication is
// lock-free so concurrent route completion cannot perturb the timed stages.
func Dump(w io.Writer) {
	s := active.Load()
	if s == nil {
		return
	}
	count := min(int(s.next.Load()), eventCapacity)
	events := make([]*event, 0, count)
	for i := range count {
		if current := s.events[i].Load(); current != nil {
			events = append(events, current)
		}
	}
	sort.Slice(events, func(i, j int) bool { return events[i].at.Before(events[j].at) })
	for _, current := range events {
		fmt.Fprintf(w, "PHASE\t%d\t%d\t%s\t%s\n",
			current.at.UnixNano(), current.at.Sub(s.started).Nanoseconds(),
			current.label, current.detail)
	}
	if dropped := int(s.next.Load()) - eventCapacity; dropped > 0 {
		fmt.Fprintf(w, "PHASE_DROPPED\t%d\n", dropped)
	}
	s.countMu.Lock()
	labels := make([]string, 0, len(s.counters))
	for label := range s.counters {
		labels = append(labels, label)
	}
	sort.Strings(labels)
	for _, label := range labels {
		fmt.Fprintf(w, "COUNT\t%s\t%d\t%d\n", label, s.counters[label].ops, s.counters[label].bytes)
	}
	s.countMu.Unlock()
}
