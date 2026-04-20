# Proofs: Task 08 - Validate and capture proof artifacts

## Planned evidence

- Output of `sqlc generate` — no errors.
- Output of `go build ./...` from repo root — exit 0.
- Output of `go vet ./...` from repo root — exit 0.
- Output of `go test ./discord/... -v -run "Portfolio|Custom"` — all resolver tests pass.
- Output of `go test ./apps/scheduler/... -v -run "Fanout|OnTriggered"` — fan-out tests pass.
- Output of `go test ./...` from repo root — full suite exits 0 (no regressions).
- Output of `git diff --stat src/services/` — no Python changes.
- Manual smoke-test log: screenshots or text transcript showing one embed arriving on each of economy, portfolio, and custom-channel webhooks for a single triggered alert; and a dedup scenario where a shared URL receives one embed, not two.
- Checked-off Definition-of-done list from `05-validation-scheduler-portfolio-custom-fanout.md`.

## Completion notes

(Fill in after implementation)
