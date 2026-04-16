# Proofs: Task 05 - Update barrel export

## Planned evidence

- `apps/ui/web/app/scanner/_components/index.ts` — unchanged or updated.

## Completion notes

### Barrel unchanged

`ScannerResultsToolbar` and `scannerResultsColumns` are internal implementation details consumed
only by `ScannerResults.tsx`. They are not needed by `page.tsx`.

The existing barrel (`index.ts`) already exports `ScannerResults` and `scanMatchesToCsv` from
`./ScannerResults`, which is the only entry point `page.tsx` depends on. No changes required.

```ts
// index.ts (unchanged)
export { ScannerFilters, type PortfolioOption } from "./ScannerFilters";
export { ScannerConditionSection } from "./ScannerConditionSection";
export { ScannerResults, scanMatchesToCsv } from "./ScannerResults";
```
