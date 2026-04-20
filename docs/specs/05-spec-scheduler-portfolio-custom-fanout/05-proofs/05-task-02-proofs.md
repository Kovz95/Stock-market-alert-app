# Proofs: Task 02 - Implement `PortfolioResolver` in `discord/`

## Planned evidence

- `ls discord/portfolios.go` — file present.
- `grep -n "type PortfolioResolver\|func NewPortfolioResolver\|ResolveWebhooks\|DefaultResolverTTL" discord/portfolios.go` — key symbols present.
- Output of `go build ./discord/...` — exits 0.
- Output of `go vet ./discord/...` — exits 0.
- A short inline note listing the cache fields and the mutex-protected reload path (matches the spec's Caching contract).

## Completion notes

(Fill in after implementation)
