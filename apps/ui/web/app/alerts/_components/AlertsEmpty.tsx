type AlertsEmptyProps = {
  /** When set, show "no results for search" copy instead of "no alerts configured". */
  searchQuery?: string;
};

export function AlertsEmpty({ searchQuery }: AlertsEmptyProps) {
  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-4">Alerts</h1>
      <p className="text-muted-foreground">
        {searchQuery
          ? "No alerts match your search. Try a different term or clear the search."
          : "No alerts configured yet."}
      </p>
    </div>
  );
}
