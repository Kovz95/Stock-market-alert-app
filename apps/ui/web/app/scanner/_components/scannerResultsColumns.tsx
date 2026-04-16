import type { ColumnDef, FilterFn } from "@tanstack/react-table";
import type { ScanMatch } from "../../../../../../gen/ts/price/v1/price";

const facetedFilterFn: FilterFn<ScanMatch> = (row, columnId, filterValue: string[]) => {
  if (!filterValue || filterValue.length === 0) return true;
  return filterValue.includes(String(row.getValue(columnId)));
};
facetedFilterFn.autoRemove = (val: unknown) => !val || (Array.isArray(val) && val.length === 0);

export function getScannerColumns(hasMatchDates: boolean): ColumnDef<ScanMatch>[] {
  const columns: ColumnDef<ScanMatch>[] = [
    {
      accessorKey: "ticker",
      header: "Ticker",
      enableSorting: true,
      enableColumnFilter: true,
    },
    {
      accessorKey: "name",
      header: "Name",
      enableSorting: true,
      enableColumnFilter: true,
    },
  ];

  if (hasMatchDates) {
    columns.push({
      accessorKey: "matchDate",
      header: "Match Date",
      enableSorting: true,
      enableColumnFilter: true,
    });
  }

  columns.push(
    {
      accessorKey: "exchange",
      header: "Exchange",
      enableSorting: true,
      enableColumnFilter: true,
      filterFn: facetedFilterFn,
    },
    {
      accessorKey: "country",
      header: "Country",
      enableSorting: true,
      enableColumnFilter: true,
      filterFn: facetedFilterFn,
    },
    {
      accessorKey: "assetType",
      header: "Type",
      enableSorting: true,
      enableColumnFilter: true,
      filterFn: facetedFilterFn,
    },
    {
      accessorKey: "price",
      header: "Price",
      enableSorting: true,
      enableColumnFilter: false,
      cell: ({ getValue }) => {
        const v = getValue();
        return typeof v === "number" ? v.toFixed(2) : String(v ?? "");
      },
    },
    {
      accessorKey: "rbicsSector",
      header: "RBICS Sector",
      enableSorting: true,
      enableColumnFilter: true,
      filterFn: facetedFilterFn,
    },
    {
      accessorKey: "rbicsIndustry",
      header: "RBICS Industry",
      enableSorting: true,
      enableColumnFilter: true,
      filterFn: facetedFilterFn,
    },
  );

  return columns;
}
