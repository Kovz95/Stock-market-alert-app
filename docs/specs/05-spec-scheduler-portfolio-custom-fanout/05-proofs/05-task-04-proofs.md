# Proofs: Task 04 - Wire resolvers into the scheduler's `Common` struct

## Planned evidence

- `git diff apps/scheduler/internal/handler/common.go` showing the new `Portfolios` and `CustomChannels` fields, the `seen` dedup map, the `addURL` closure, and the two new resolver calls around the existing economy dispatch.
- `git diff apps/scheduler/cmd/` showing the resolver construction in the scheduler's startup path (with `pool`, `discord.DefaultResolverTTL`, and the shared logger).
- Output of `go build ./apps/scheduler/...` — exits 0.
- `grep -n "seen\s*:=" apps/scheduler/internal/handler/common.go` — confirms dedup is at the call site, not inside a resolver.

## Completion notes

(Fill in after implementation)
