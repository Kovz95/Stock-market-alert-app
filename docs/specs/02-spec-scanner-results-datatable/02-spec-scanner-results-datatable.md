# Spec: Scanner Results DataTable (02)

## Goal

The scanner results are currently rendered with a plain shadcn `Table`, a single global search input, and a hard 500-row display cap. This makes it difficult to isolate results by specific exchange, country, asset type, or sector without manually reading through rows. Replace `ScannerResults.tsx` with a TanStack Table-backed shadcn DataTable that supports per-column filtering, user-controlled sorting, and pagination — removing the 500-row cap while keeping CSV export intact.

## Scope

### In scope
- Create `apps/ui/web/components/ui/data-table.tsx` — a reusable shadcn DataTable wrapper around `@tanstack/react-table` v8 that supports: column definitions, column visibility, column filters (text and faceted), sorting, and pagination.
- Replace `ScannerResults.tsx` with a new implementation that uses `data-table.tsx` and defines `ColumnDef[]` for all existing scanner result columns.
- Per-column filters:
  - **Text filter**: Ticker, Name, Match Date
  - **Faceted (multi-select) filter**: Exchange, Country, Asset Type, RBICS Sector, RBICS Industry
- Sortable column headers for all columns; default initial sort: Match Date desc → Ticker asc.
- Pagination (configurable page size, default 50 rows/page).
- Remove the existing global search `Input` — column filters replace it.
- Retain the "Download CSV" button wired to the **full unfiltered** `matches` array (unchanged behavior).
- Retain the "Scanning…" / progress / `matches.length` count display (unchanged behavior).
- The rest of the scanner page (`page.tsx`, `ScannerFilters.tsx`, `ScannerConditionSection.tsx`, atoms, server actions) is **not modified**.

### Out of scope
- Price range / numeric column filtering.
- Server-side filtering or pagination (all filtering is client-side).
- Virtualization / windowed rendering.
- Column reordering or user-defined column visibility toggles.
- Applying DataTable to any page other than Scanner Results.
- Changes to `ScanMatch` proto type or server actions.

## Acceptance criteria

1. **DataTable component exists**
   - `apps/ui/web/components/ui/data-table.tsx` is present and exports a `DataTable` component.
   - Running `pnpm --filter web typecheck` exits 0 with no errors introduced by this file.

2. **Scanner Results renders via DataTable**
   - `ScannerResults.tsx` no longer imports from `@/components/ui/table` directly; it renders via `DataTable`.
   - Running `pnpm --filter web typecheck` exits 0 after the replacement.

3. **Column filters are present for correct columns**
   - The rendered table toolbar contains text filter inputs for: Ticker, Name, and (when `hasMatchDates` is true) Match Date.
   - The rendered table toolbar contains faceted filter controls (multi-select or dropdown) for: Exchange, Country, Asset Type, RBICS Sector, RBICS Industry.
   - Price column has no filter control in the toolbar.

4. **Column sorting works**
   - Every column header is clickable and cycles through: unsorted → ascending → descending.
   - On initial render with match dates present, rows are sorted by Match Date descending, then Ticker ascending.

5. **Pagination is present**
   - A pagination control (prev/next, page size selector) is rendered below the table.
   - Default page size is 50 rows.
   - The 500-row hard cap in the original `ScannerResults.tsx` is removed; all rows are accessible via pagination.

6. **CSV export uses unfiltered data**
   - The "Download CSV" button calls `scanMatchesToCsv` with the full `matches` prop (not the filtered/paginated subset).
   - `scanMatchesToCsv` is unchanged.

7. **No regressions on zero results**
   - When `matches` is an empty array, the component renders the "0 matches" state (no table, no crash).

8. **Global search input removed**
   - The `<Input placeholder="Search ticker or name..." />` that existed in the old `ScannerResults.tsx` is no longer rendered.

## Conventions

- Use `@tanstack/react-table` v8 API (`useReactTable`, `getCoreRowModel`, `getFilteredRowModel`, `getSortedRowModel`, `getPaginationRowModel`, `getFacetedRowModel`, `getFacetedUniqueValues`).
- Follow the shadcn/ui DataTable documentation pattern for component structure (toolbar, table body, pagination).
- The `data-table.tsx` component must be generic (`TData`, `TValue`) so it can be reused for other tables in the future.
- Do not introduce any new npm dependencies; `@tanstack/react-table` is already present.
- The `matchDate` column must remain conditionally rendered (only when `hasMatchDates` is true), matching existing behavior.
- All existing `ScannerResultsProps` must remain satisfied; the component signature must not break the call site in `page.tsx`.
