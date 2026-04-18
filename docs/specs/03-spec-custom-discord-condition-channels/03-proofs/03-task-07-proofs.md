# Proofs: Task 07 - Validate and capture proof artifacts

## Planned evidence

- Full captured output of:
  - `pnpm --filter web typecheck` (must exit 0).
  - `pnpm --filter web lint` (must exit 0, or document any pre-existing warnings).
  - `go test ./apps/grpc/discord_service/...` (must exit 0).
- Screenshots or HTTP logs from the manual checks listed in `03-validation-custom-discord-condition-channels.md`:
  1. New channel appears in both React UI and Streamlit UI.
  2. Test embed arrives in Discord with the stored condition in the body.
  3. Toggling enabled in React is reflected in Streamlit.
  4. Delete in React removes the entry in Streamlit.
  5. Invalid condition: toast error screenshot + evidence no row was persisted.
  6. Duplicate name: toast error screenshot.
  7. Price-level channel created and persisted with literal `"price_level"`.
  8. End-to-end alert delivery: a webhook.site (or Discord) capture confirming the custom channel received an alert when the condition matched.
  9. Price-level channel receives an alert whose condition is `Close[-1] < 200`.
- A final checklist copy from the validation file's "Definition of done" with each box ticked.

## Completion notes

(Fill in after implementation)
