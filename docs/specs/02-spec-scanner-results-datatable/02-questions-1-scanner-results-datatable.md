# Questions: Scanner Results DataTable (02)

## Resolved

| # | Question | Resolution |
|---|----------|------------|
| R1 | Is `@tanstack/react-table` already installed? | Yes — `^8.21.3` in `apps/ui/web/package.json`. No new dependency needed. |
| R2 | Does a shared `data-table.tsx` exist in `components/ui/`? | No. Must be created as part of this spec. |
| R3 | Which component is being replaced? | `apps/ui/web/app/scanner/_components/ScannerResults.tsx` only. The rest of the scanner page is unchanged. |
| R4 | What columns exist in `ScanMatch`? | ticker, name, matchDate (optional), exchange, country, assetType, price, rbicsSector, rbicsIndustry. |
| R5 | What filtering style fits each column? | Text filter: Ticker, Name. Faceted (multi-select dropdown) filter: Exchange, Country, AssetType, RBICS Sector, RBICS Industry. Match Date: text filter. Price: no column filter (numeric range is out of scope). |
| R6 | Should sorting be user-controlled? | Yes — clickable column headers (standard TanStack behavior). Default sort: matchDate desc, then ticker asc (preserves existing behavior). |
| R7 | Where does the `data-table.tsx` component live? | `apps/ui/web/components/ui/data-table.tsx` — shared component following shadcn convention. |
| R8 | Does the CSV download stay? | Yes — unchanged behavior, wired to the full unfiltered results. |

## Open

| # | Question | Impact |
|---|----------|--------|
| O1 | Pagination vs. keep 500-row client cap? | Default spec targets **pagination** (e.g. 50 rows/page) because DataTable naturally supports it and removes the arbitrary 500-row hard cap. If virtualization is preferred, this spec should be revised before implementation starts. |
| O2 | Remove global search input or keep alongside column filters? | Default spec **removes the global search input** and replaces it with per-column filters, which are more precise. If global search should be retained, add it as a toolbar input wired to a global filter in TanStack Table. |
