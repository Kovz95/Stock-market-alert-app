"use client";

import * as React from "react";
import type { Table } from "@tanstack/react-table";
import { ChevronDownIcon } from "lucide-react";
import type { ScanMatch } from "../../../../../../gen/ts/price/v1/price";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface FacetedFilterProps {
  table: Table<ScanMatch>;
  columnId: string;
  label: string;
}

function FacetedFilter({ table, columnId, label }: FacetedFilterProps) {
  const column = table.getColumn(columnId);
  if (!column) return null;

  const facetedValues = column.getFacetedUniqueValues();
  const options = Array.from(facetedValues.keys())
    .filter((v) => v != null && v !== "")
    .map(String)
    .sort();
  const filterValue = (column.getFilterValue() as string[] | undefined) ?? [];

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm" className="h-8 gap-1 text-xs">
          {label}
          {filterValue.length > 0 && (
            <span className="rounded-full bg-primary px-1.5 py-0.5 text-[10px] leading-none text-primary-foreground">
              {filterValue.length}
            </span>
          )}
          <ChevronDownIcon className="h-3 w-3" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        {options.map((option) => (
          <DropdownMenuCheckboxItem
            key={option}
            checked={filterValue.includes(option)}
            onCheckedChange={(checked) => {
              const next = checked
                ? [...filterValue, option]
                : filterValue.filter((v) => v !== option);
              column.setFilterValue(next.length > 0 ? next : undefined);
            }}
          >
            {option}
          </DropdownMenuCheckboxItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface ScannerResultsToolbarProps {
  table: Table<ScanMatch>;
}

export function ScannerResultsToolbar({ table }: ScannerResultsToolbarProps) {
  const matchDateColumn = table.getColumn("matchDate");

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Input
        placeholder="Filter ticker..."
        className="h-8 w-32 text-xs"
        value={(table.getColumn("ticker")?.getFilterValue() as string) ?? ""}
        onChange={(e) =>
          table.getColumn("ticker")?.setFilterValue(e.target.value || undefined)
        }
      />
      <Input
        placeholder="Filter name..."
        className="h-8 w-40 text-xs"
        value={(table.getColumn("name")?.getFilterValue() as string) ?? ""}
        onChange={(e) =>
          table.getColumn("name")?.setFilterValue(e.target.value || undefined)
        }
      />
      {matchDateColumn && (
        <Input
          placeholder="Filter date..."
          className="h-8 w-32 text-xs"
          value={(matchDateColumn.getFilterValue() as string) ?? ""}
          onChange={(e) =>
            matchDateColumn.setFilterValue(e.target.value || undefined)
          }
        />
      )}
      <FacetedFilter table={table} columnId="exchange" label="Exchange" />
      <FacetedFilter table={table} columnId="country" label="Country" />
      <FacetedFilter table={table} columnId="assetType" label="Type" />
      <FacetedFilter table={table} columnId="rbicsSector" label="RBICS Sector" />
      <FacetedFilter table={table} columnId="rbicsIndustry" label="RBICS Industry" />
      <Button
        variant="ghost"
        size="sm"
        className="h-8 text-xs"
        onClick={() => table.resetColumnFilters()}
      >
        Reset filters
      </Button>
    </div>
  );
}
