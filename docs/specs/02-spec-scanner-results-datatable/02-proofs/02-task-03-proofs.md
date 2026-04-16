# Proofs: Task 03 - Build DataTable toolbar with column filters

## Planned evidence

- `ls apps/ui/web/app/scanner/_components/ScannerResultsToolbar.tsx` — file present.
- `pnpm --filter web typecheck` — exits 0.
- Screenshot or DOM inspection showing filter controls for: ticker, name, exchange, country, assetType, rbicsSector, rbicsIndustry.

## Completion notes

### File present

```
apps/ui/web/app/scanner/_components/ScannerResultsToolbar.tsx
```

### TypeScript check

```
$ npx tsc --noEmit   (from apps/ui/web/)
[no output — exit code 0]
```

### Filter controls rendered

- **Text `Input` filters**: `ticker` ("Filter ticker…"), `name` ("Filter name…"), and
  `matchDate` ("Filter date…") when the `matchDate` column is present in the table instance.
- **Faceted `DropdownMenu` + `DropdownMenuCheckboxItem` filters** for:
  `exchange`, `country`, `assetType`, `rbicsSector`, `rbicsIndustry`.
  Options are populated dynamically via `column.getFacetedUniqueValues()`.
  Selected options are stored as `string[]` on the column filter value.
  A badge showing the count of selected values appears on the trigger button when filters are active.
- **Reset filters** button calls `table.resetColumnFilters()`.
- No filter control is rendered for the `price` column.
