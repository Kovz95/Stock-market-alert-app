# Proofs: Task 02 - Implement the Go `DiscordConfigService` methods

## Planned evidence

- Listing of new/modified Go files in `apps/grpc/discord_service/` and any `database/` helpers added.
- `grep` output confirming all five methods are defined on `*Server` with no `codes.Unimplemented` returns for custom RPCs.
- Go test output (`go test ./apps/grpc/discord_service/...`) demonstrating:
  - Create → List round-trip.
  - Duplicate-name rejection.
  - Invalid condition rejection via `expr.Parse`.
  - `price_level` accepted without parse.
  - Update applies only set fields.
  - Delete removes the entry; second Delete returns not-found.
- Evidence that the same document read by Python is written by Go (e.g. a Postgres `app_documents` row dump or a shared-file cat/diff before and after Go mutations).

## Completion notes

(Fill in after implementation)
