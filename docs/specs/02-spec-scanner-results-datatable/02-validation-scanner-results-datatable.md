# Validation: Scanner Results DataTable (02)

## Automated verification

From repository root (`apps/ui/web/`):

```bash
# AC1 + AC2: DataTable component exists and no type errors
pnpm --filter web typecheck
# Expected: exit code 0, no errors

# AC1: File exists
ls apps/ui/web/components/ui/data-table.tsx
# Expected: file listed with non-zero size

# AC2: ScannerResults no longer imports from @/components/ui/table directly
# Expected: no output (import removed)
Select-String -Path "apps/ui/web/app/scanner/_components/ScannerResults.tsx" -Pattern "from.*@/components/ui/table"

# AC5: 500-row slice cap is removed
# Expected: no output (cap removed)
Select-String -Path "apps/ui/web/app/scanner/_components/ScannerResults.tsx" -Pattern "slice\(0, 500\)"

# AC6: CSV export still uses full matches prop
# Expected: line containing scanMatchesToCsv(matches) — not filtered/paginated data
Select-String -Path "apps/ui/web/app/scanner/_components/ScannerResults.tsx" -Pattern "scanMatchesToCsv\(matches\)"

# AC8: Global search input removed
# Expected: no output (input removed)
Select-String -Path "apps/ui/web/app/scanner/_components/ScannerResults.tsx" -Pattern "Search ticker or name"
```

**Expected outcomes:**
- `typecheck`: exits 0
- `data-table.tsx`: file present
- Direct `@/components/ui/table` import in `ScannerResults.tsx`: no match
- `slice(0, 500)`: no match
- `scanMatchesToCsv(matches)`: one match
- "Search ticker or name": no match

## Traceability

- Feature spec: `02-spec-scanner-results-datatable.md`
- Task breakdown: `02-tasks-scanner-results-datatable.md`
- Questions and decisions: `02-questions-1-scanner-results-datatable.md`
- Per-task evidence: `02-proofs/02-task-NN-proofs.md`
- Upstream specs: none

## Manual checks

1. Run the dev server (`pnpm --filter web dev`), navigate to `/scanner`, run a scan.
2. Verify column headers are clickable and toggle sort order (asc/desc indicator visible).
3. Type in the Ticker text filter — rows reduce to matches only.
4. Open the Exchange faceted filter — options populated from scan results; selecting one filters rows.
5. Open the RBICS Sector faceted filter — same behavior.
6. Advance to page 2 via pagination; confirm rows advance correctly.
7. Change page size to 100; confirm more rows appear per page.
8. Click "Reset filters" — all filters clear, full result set restored.
9. Click "Download CSV" — file downloads with all `matches.length` rows (not just the current page/filter view).
10. Run a scan returning 0 matches — verify "0 matches" state renders without errors.

## Definition of done

- [ ] AC1: `data-table.tsx` exists; `pnpm --filter web typecheck` exits 0
- [ ] AC2: `ScannerResults.tsx` uses `DataTable`; typecheck exits 0
- [ ] AC3: Text filters for Ticker, Name, Match Date; faceted filters for Exchange, Country, AssetType, Sector, Industry
- [ ] AC4: Column headers sortable; default sort is Match Date desc → Ticker asc
- [ ] AC5: Pagination present, default 50 rows/page, 500-row cap removed
- [ ] AC6: CSV download uses full unfiltered `matches`
- [ ] AC7: Zero-results state renders correctly
- [ ] AC8: Global search input removed
- [ ] Proof artifacts contain real command output, not placeholders
