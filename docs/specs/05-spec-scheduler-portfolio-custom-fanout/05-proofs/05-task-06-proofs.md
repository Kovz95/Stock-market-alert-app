# Proofs: Task 06 - Unit tests for `CustomChannelResolver`

## Planned evidence

- `ls discord/custom_test.go` — file present.
- Output of `go test ./discord/... -v -run "TestCustomChannelResolver"` showing all eleven subtests pass:
  - `TestCustomChannelResolver_Empty`
  - `TestCustomChannelResolver_Enabled`
  - `TestCustomChannelResolver_Disabled`
  - `TestCustomChannelResolver_LegacySchemaSkipped`
  - `TestCustomChannelResolver_NormalizationWhitespace`
  - `TestCustomChannelResolver_NormalizationOperators`
  - `TestCustomChannelResolver_NormalizationCase`
  - `TestCustomChannelResolver_PriceLevelMatch`
  - `TestCustomChannelResolver_PriceLevelCaseInsensitive`
  - `TestCustomChannelResolver_ExactMatchNotPartial`
  - `TestCustomChannelResolver_CacheHit` / `TestCustomChannelResolver_CacheReload`

## Completion notes

(Fill in after implementation)
