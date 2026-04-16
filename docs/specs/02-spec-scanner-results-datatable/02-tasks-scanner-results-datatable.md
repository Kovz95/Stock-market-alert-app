# Tasks: Scanner Results DataTable (02)

## Task 01 - Create shared DataTable component

- Create `apps/ui/web/components/ui/data-table.tsx`.
- Use `@tanstack/react-table` v8: `useReactTable` with `getCoreRowModel`, `getFilteredRowModel`, `getSortedRowModel`, `getPaginationRowModel`, `getFacetedRowModel`, `getFacetedUniqueValues`.
- Component signature: `DataTable<TData, TValue>({ columns, data, initialSorting? })`.
- Render a toolbar slot (accept as a render prop or child) for filter controls above the table.
- Render the shadcn `Table` / `TableHeader` / `TableBody` / `TableRow` / `TableHead` / `TableCell` primitives from `@/components/ui/table`.
- Render a pagination row below the table: Previous / Next buttons (shadcn `Button`), current page indicator, and a page size `Select` with options [25, 50, 100, 200].
- Export: `DataTable` (default or named), plus helper types `ColumnDef` re-exported from `@tanstack/react-table` for convenience.

**Proof:** 02-proofs/02-task-01-proofs.md

## Task 02 - Define ScanMatch column definitions

- Create `apps/ui/web/app/scanner/_components/scannerResultsColumns.tsx` (new file).
- Define and export `getScannerColumns(hasMatchDates: boolean): ColumnDef<ScanMatch>[]`.
- Include all columns: ticker, name, matchDate (only when `hasMatchDates`), exchange, country, assetType, price, rbicsSector, rbicsIndustry.
- Each column specifies: `accessorKey`, `header` (string), `enableSorting: true`, and `enableColumnFilter: true` (or `false` for price).
- Price column: `cell` renderer that calls `.toFixed(2)` when numeric.

**Proof:** 02-proofs/02-task-02-proofs.md

## Task 03 - Build DataTable toolbar with column filters

- Create `apps/ui/web/app/scanner/_components/ScannerResultsToolbar.tsx` (new file).
- Accept `table: Table<ScanMatch>` as a prop.
- Render text `Input` filter controls for columns: `ticker`, `name`, and `matchDate` (conditional on column existence).
- Render faceted filter controls (multi-select popover using shadcn `Popover` + `Command`, or `DropdownMenu` + `Checkbox`) for: `exchange`, `country`, `assetType`, `rbicsSector`, `rbicsIndustry`.
  - Use `table.getColumn(id)?.getFacetedUniqueValues()` to populate options dynamically from the current dataset.
- Each filter control calls `column.setFilterValue(...)` on change.
- Include a "Reset filters" button that calls `table.resetColumnFilters()`.

**Proof:** 02-proofs/02-task-03-proofs.md

## Task 04 - Replace ScannerResults with DataTable implementation

- Rewrite `apps/ui/web/app/scanner/_components/ScannerResults.tsx`.
- Remove the old `Table` import, manual `search` state, and `filtered.slice(0, 500)` cap.
- Use `getScannerColumns(hasMatchDates)` from Task 02.
- Render `<DataTable columns={columns} data={sorted} initialSorting={[{ id: "matchDate", desc: true }]} toolbar={(table) => <ScannerResultsToolbar table={table} />} />`.
- Retain: result count header, scanning/progress display, "Download CSV" button wired to the full `matches` prop.
- Retain the zero-results early return ("0 matches" state).
- Keep `scanMatchesToCsv` export in this file (unchanged).
- `ScannerResultsProps` signature must not change.

**Proof:** 02-proofs/02-task-04-proofs.md

## Task 05 - Update barrel export

- Update `apps/ui/web/app/scanner/_components/index.ts` to export any new symbols needed by `page.tsx` (verify no new exports are required; `ScannerResults` and `scanMatchesToCsv` already exported).
- If `ScannerResultsToolbar` or `scannerResultsColumns` need to be exported, add them; otherwise leave the barrel unchanged.

**Proof:** 02-proofs/02-task-05-proofs.md

## Task 06 - Validate and capture proof artifacts

- Run `pnpm --filter web typecheck` and capture output (must exit 0).
- Run `pnpm --filter web lint` (or equivalent) and capture output.
- Manually verify in the browser: column filter inputs appear, faceted filters populate from scan results, sorting works on header click, pagination controls appear, CSV downloads full unfiltered data.
- Confirm all acceptance criteria from `02-spec-scanner-results-datatable.md` pass.
- Fill in all proof files with real command output.

**Proof:** 02-proofs/02-task-06-proofs.md
