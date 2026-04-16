# Proofs: Task 02 - Define ScanMatch column definitions

## Planned evidence

- `ls apps/ui/web/app/scanner/_components/scannerResultsColumns.tsx` — file present.
- `pnpm --filter web typecheck` — exits 0.

## Completion notes

### File present

```
apps/ui/web/app/scanner/_components/scannerResultsColumns.tsx
```

### TypeScript check

```
$ npx tsc --noEmit   (from apps/ui/web/)
[no output — exit code 0]
```

### Implementation notes

- Exports `getScannerColumns(hasMatchDates: boolean): ColumnDef<ScanMatch>[]`.
- Columns included: `ticker`, `name`, `matchDate` (only when `hasMatchDates`), `exchange`,
  `country`, `assetType`, `price`, `rbicsSector`, `rbicsIndustry`.
- All columns have `enableSorting: true`; price has `enableColumnFilter: false`, all others `true`.
- Faceted columns (`exchange`, `country`, `assetType`, `rbicsSector`, `rbicsIndustry`) use a custom
  `filterFn` that checks `filterValue.includes(cellValue)` for multi-select array filter values.
- Price `cell` renderer calls `.toFixed(2)` when value is numeric.
