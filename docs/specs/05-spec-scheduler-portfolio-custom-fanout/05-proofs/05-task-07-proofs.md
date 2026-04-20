# Proofs: Task 07 - Integration test for `onTriggered` fan-out + dedup

## Planned evidence

- `ls apps/scheduler/internal/handler/common_fanout_test.go` — file present.
- Output of `go test ./apps/scheduler/... -v -run "Fanout|OnTriggered"` showing both cases pass:
  - Standard 3-destination: economy `W1`, portfolio `W2`, custom channel `W3` → three `Accum.Add` calls, distinct URLs, same embed pointer.
  - Dedup case: economy `W1`, portfolio also `W1`, custom channel `W2` → exactly two `Accum.Add` calls (W1, W2).
- A short inline snippet from the test showing the `Accum` mock's recorded calls confirming embed pointer equality across destinations.

## Completion notes

(Fill in after implementation)
