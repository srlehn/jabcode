// This file exists only to keep the committed capture fixtures out of
// the parent module zip: module zips exclude any subtree carrying its
// own go.mod, and without that exclusion these captures push the zip
// past Go's 500 MB module limit, so consumers cannot go get the
// library at all. There is no Go code below this directory; clones
// still receive the files and the harnesses read them from disk.
module github.com/srlehn/jabcode/testdata/highcolor_capture

go 1.26.0
