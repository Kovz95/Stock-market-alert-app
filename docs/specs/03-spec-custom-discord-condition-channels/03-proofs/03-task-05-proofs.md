# Proofs: Task 05 - Build `/discord/custom` page

## Planned evidence

- `ls` of `apps/ui/web/app/discord/custom/` showing `page.tsx` and `_components/` with component files.
- `grep` output confirming imports from `apps/ui/web/app/alerts/add/_components/` (proves `ConditionBuilder` reuse, not duplication).
- Screenshot of `/discord/custom` page in dev mode showing:
  - Empty state when no channels exist.
  - Create form with the radio toggle ("Specific condition" vs "Any price level").
  - A populated list after creating one specific-condition channel and one `price_level` channel.
- Screenshot/log showing a Sonner error toast for an invalid condition submission.
- `pnpm --filter web typecheck` output (exit 0).

## Completion notes

(Fill in after implementation)
