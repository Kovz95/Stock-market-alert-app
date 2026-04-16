# Proofs: Task 01 - Create shared DataTable component

## Planned evidence

- `ls apps/ui/web/components/ui/data-table.tsx` — file present with non-zero size.
- `pnpm --filter web typecheck` output — exits 0 after file created.

## Completion notes

### File present

```
apps/ui/web/components/ui/data-table.tsx  (5303 bytes)
```

### TypeScript check

```
$ npx tsc --noEmit   (from apps/ui/web/)
[no output — exit code 0]
```

TypeScript exits 0 with no errors introduced by this file.

### Implementation notes

- Generic `DataTable<TData, TValue>` component using `useReactTable` with all six row models:
  `getCoreRowModel`, `getFilteredRowModel`, `getSortedRowModel`, `getPaginationRowModel`,
  `getFacetedRowModel`, `getFacetedUniqueValues`.
- Accepts `columns`, `data`, `initialSorting?` (defaults to `[]`), and `toolbar` render prop.
- Renders shadcn `Table` / `TableHeader` / `TableBody` / `TableRow` / `TableHead` / `TableCell`.
- Pagination row: Previous / Next buttons, page indicator, page-size `Select` with [25, 50, 100, 200].
- Default page size is 50 (set via `initialState.pagination.pageSize`).
- Exports `DataTable` (named), `ColumnDef`, and `Table` re-exported from `@tanstack/react-table`.
