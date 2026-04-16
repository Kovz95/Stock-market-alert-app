# Proofs: Task 04 - Replace ScannerResults with DataTable implementation

## Planned evidence

- `ScannerResults.tsx` no longer imports from `@/components/ui/table` directly.
- `pnpm --filter web typecheck` — exits 0.

## Completion notes

### Direct table import removed

`ScannerResults.tsx` no longer imports from `@/components/ui/table`. It imports:
- `DataTable` from `@/components/ui/data-table`
- `getScannerColumns` from `./scannerResultsColumns`
- `ScannerResultsToolbar` from `./ScannerResultsToolbar`
- `Button` from `@/components/ui/button`

### TypeScript check

```
$ npx tsc --noEmit   (from apps/ui/web/)
[no output — exit code 0]
```

### Changes from original

- Removed: manual `search` state, `filtered` memo, `filtered.slice(0, 500)` cap, global
  `<Input placeholder="Search ticker or name..." />`.
- Renders `<DataTable columns={columns} data={sorted} initialSorting={[{ id: "matchDate", desc: true }]}
  toolbar={(table) => <ScannerResultsToolbar table={table} />} />`.
- Retained: result count header, scanning/progress display, "Download CSV" button wired to full
  `matches` prop (not the filtered subset).
- Retained: zero-results early return ("0 matches" state, unchanged markup).
- Retained: `scanMatchesToCsv` export (unchanged).
- `ScannerResultsProps` signature unchanged; call site in `page.tsx` unaffected.
