# Proofs: Task 04 — Rename scanner lookback label and add help text

## Planned evidence

- Diff of `apps/ui/web/app/scanner/page.tsx` showing:
  - `<Label>` text changed from `Lookback days` to `Lookback bars`.
  - A new `<p className="text-xs text-muted-foreground">…</p>` sibling explaining bar-count semantics.
  - No changes to `min`, `max`, `onChange`, `onBlur`, or the two Jotai atoms.
- Output of:
  ```bash
  grep -n "Lookback bars\|Lookback days" apps/ui/web/app/scanner/page.tsx
  pnpm typecheck
  pnpm lint
  ```
- Screenshot of the scanner page showing the new label and help text (optional but preferred).

## Completion notes

(Fill in after implementation)
