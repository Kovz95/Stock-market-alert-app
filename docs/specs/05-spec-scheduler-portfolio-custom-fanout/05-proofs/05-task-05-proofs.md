# Proofs: Task 05 - Unit tests for `PortfolioResolver`

## Planned evidence

- `ls discord/portfolios_test.go` — file present.
- Output of `go test ./discord/... -v -run "TestPortfolioResolver"` showing all nine subtests pass:
  - `TestPortfolioResolver_Empty`
  - `TestPortfolioResolver_SingleMatch`
  - `TestPortfolioResolver_MultiMatch`
  - `TestPortfolioResolver_CaseInsensitive`
  - `TestPortfolioResolver_ExchangeSuffix`
  - `TestPortfolioResolver_DisabledExcluded`
  - `TestPortfolioResolver_BlankWebhookExcluded`
  - `TestPortfolioResolver_CacheHit`
  - `TestPortfolioResolver_CacheReload`
- A short explanation of the DB mock/real-DB pattern chosen (matching existing `discord/` tests).

## Completion notes

(Fill in after implementation)
