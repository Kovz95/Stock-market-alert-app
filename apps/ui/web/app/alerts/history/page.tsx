"use client";

import { useState, useMemo, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import {
  useStockSearch,
  useTriggerHistoryByTicker,
  usePortfolios,
  useAlertsForHistory,
} from "@/lib/hooks/useAlertHistory";
import { useAuditSummary } from "@/lib/hooks/useAudit";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { formatAuditDateTime } from "@/app/alerts/audit/_components/formatAuditDate";
import { formatCondition } from "./_components/formatCondition";
import { getAlertConditionStrings } from "./_components/alertConditionStrings";
import type { AlertData } from "@/actions/alert-actions";
import type { AuditHistoryRow } from "@/actions/alert-history-actions";
import type { StockSearchResult } from "@/actions/alert-history-actions";
import type { Portfolio } from "@/actions/alert-history-actions";
import { SearchIcon, FolderIcon, ChevronDownIcon } from "lucide-react";

const POPULAR_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"];

function TriggerHistoryList({
  rows,
  showAll,
  daysBack,
}: {
  rows: AuditHistoryRow[];
  showAll: boolean;
  daysBack: number;
}) {
  const displayRows = useMemo(() => {
    const filtered = rows.filter((r) => {
      if (daysBack <= 0) return true;
      const ts = r.timestamp ? new Date(r.timestamp).getTime() : 0;
      const cutoff = Date.now() - daysBack * 24 * 60 * 60 * 1000;
      return ts >= cutoff;
    });
    return showAll ? filtered.slice(0, 50) : filtered.filter((r) => r.alertTriggered).slice(0, 20);
  }, [rows, showAll, daysBack]);

  const triggerCount = useMemo(
    () => rows.filter((r) => r.alertTriggered).length,
    [rows]
  );
  const uniqueAlerts = useMemo(
    () => new Set(rows.map((r) => r.alertName || r.alertId)).size,
    [rows]
  );
  const lastTrigger = useMemo(() => {
    const triggered = rows.filter((r) => r.alertTriggered);
    if (triggered.length === 0) return null;
    const dates = triggered
      .map((r) => (r.timestamp ? new Date(r.timestamp).getTime() : 0))
      .filter(Boolean);
    return dates.length ? new Date(Math.max(...dates)) : null;
  }, [rows]);
  const daysSinceLastTrigger = lastTrigger
    ? Math.floor((Date.now() - lastTrigger.getTime()) / (24 * 60 * 60 * 1000))
    : null;

  return (
    <>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        <Card size="sm">
          <CardContent className="pt-3">
            <p className="text-muted-foreground text-xs">Total triggers</p>
            <p className="text-lg font-semibold">{triggerCount}</p>
          </CardContent>
        </Card>
        <Card size="sm">
          <CardContent className="pt-3">
            <p className="text-muted-foreground text-xs">Total evaluations</p>
            <p className="text-lg font-semibold">{rows.length}</p>
          </CardContent>
        </Card>
        <Card size="sm">
          <CardContent className="pt-3">
            <p className="text-muted-foreground text-xs">Unique alerts</p>
            <p className="text-lg font-semibold">{uniqueAlerts}</p>
          </CardContent>
        </Card>
        <Card size="sm">
          <CardContent className="pt-3">
            <p className="text-muted-foreground text-xs">Days since last trigger</p>
            <p className="text-lg font-semibold">
              {daysSinceLastTrigger != null ? daysSinceLastTrigger : "—"}
            </p>
          </CardContent>
        </Card>
      </div>
      <div className="space-y-2">
        {displayRows.map((row, idx) => (
          <details
            key={row.id ?? `${row.timestamp}-${idx}`}
            className="border rounded-md overflow-hidden"
          >
            <summary className="px-3 py-2 cursor-pointer list-none flex items-center gap-2 hover:bg-muted/50">
              <span>{row.alertTriggered ? "✅" : "❌"}</span>
              <span className="font-medium">
                {row.alertName || row.alertId?.slice(0, 8) || "Alert"}
              </span>
              <span className="text-muted-foreground text-xs">
                {row.timestamp ? formatAuditDateTime(row.timestamp) : "—"}
              </span>
              <ChevronDownIcon className="size-4 ml-auto" />
            </summary>
            <div className="px-3 pb-3 pt-0 grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm border-t">
              <div>
                <p><strong>Triggered:</strong> {row.alertTriggered ? "Yes ✅" : "No ❌"}</p>
                <p><strong>Source:</strong> {row.evaluationType ?? "—"}</p>
                <p><strong>Execution time:</strong> {row.executionTimeMs != null ? `${row.executionTimeMs} ms` : "N/A"}</p>
              </div>
              <div>
                <p className="font-medium">Condition details</p>
                <pre className="text-xs whitespace-pre-wrap bg-muted p-2 rounded mt-1">
                  {formatCondition(row.triggerReason)}
                </pre>
              </div>
            </div>
          </details>
        ))}
      </div>
    </>
  );
}

function AlertHistoryContent() {
  const searchParams = useSearchParams();
  const urlTicker = searchParams.get("ticker") ?? "";
  const urlTab = searchParams.get("tab") ?? "search";

  const [activeTab, setActiveTab] = useState(urlTab === "browse" ? "browse" : "search");
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);
  const [showAllEvaluations, setShowAllEvaluations] = useState(false);
  const [daysBack, setDaysBack] = useState(30);
  const [hasSearched, setHasSearched] = useState(false);

  // Browse tab state
  const [browseDays, setBrowseDays] = useState(30);
  const [triggerStatus, setTriggerStatus] = useState<"all" | "triggered" | "not_triggered">("all");
  const [selectedExchanges, setSelectedExchanges] = useState<string[]>([]);
  const [selectedActions, setSelectedActions] = useState<string[]>([]);
  const [inAnyPortfolio, setInAnyPortfolio] = useState(false);
  const [selectedPortfolioIds, setSelectedPortfolioIds] = useState<string[]>([]);
  const [selectedConditions, setSelectedConditions] = useState<string[]>([]);
  const [browseApplied, setBrowseApplied] = useState(false);

  useEffect(() => {
    if (urlTab === "search" && urlTicker) {
      setActiveTab("search");
      setSearchTerm(urlTicker);
      setSelectedTicker(urlTicker);
      setHasSearched(true);
    }
  }, [urlTab, urlTicker]);

  const { data: searchResults, isFetching: searchLoading } = useStockSearch(
    searchTerm,
    hasSearched && searchTerm.length >= 2
  );
  const tickerToShow = selectedTicker || (searchResults?.length === 1 ? searchResults[0].ticker : null);
  const { data: historyRows, isLoading: historyLoading } = useTriggerHistoryByTicker(
    tickerToShow ?? "",
    { includeAllEvaluations: showAllEvaluations, limit: 100, daysBack: 0 }
  );
  const { data: portfolios } = usePortfolios();
  const { data: allAlerts, isLoading: alertsLoading } = useAlertsForHistory();
  const { data: auditSummary } = useAuditSummary(browseDays);

  const alertsForTicker = useMemo(() => {
    if (!tickerToShow || !allAlerts) return [];
    return allAlerts.filter((a) => a.ticker === tickerToShow);
  }, [tickerToShow, allAlerts]);

  const selectedCompany = useMemo(() => {
    if (!tickerToShow || !searchResults) return null;
    return searchResults.find((r) => r.ticker === tickerToShow) ?? null;
  }, [tickerToShow, searchResults]);

  const portfolioTickers = useMemo(() => {
    const set = new Set<string>();
    const map: Record<string, string[]> = {};
    portfolios?.forEach((p) => {
      map[p.portfolioId] = p.tickers ?? [];
      p.tickers?.forEach((t) => set.add(t));
    });
    return { set, map };
  }, [portfolios]);

  const allExchanges = useMemo(() => {
    if (!allAlerts) return [];
    const s = new Set<string>();
    allAlerts.forEach((a) => a.exchange && s.add(a.exchange));
    return Array.from(s).sort();
  }, [allAlerts]);
  const allActions = useMemo(() => {
    if (!allAlerts) return [];
    const s = new Set<string>();
    allAlerts.forEach((a) => a.action && s.add(a.action));
    return Array.from(s).sort();
  }, [allAlerts]);
  const allConditionStrings = useMemo(() => {
    if (!allAlerts) return [];
    const s = new Set<string>();
    allAlerts.forEach((a) => getAlertConditionStrings(a).forEach((c) => s.add(c)));
    return Array.from(s).sort();
  }, [allAlerts]);

  const triggeredAlertIdsInPeriod = useMemo(() => {
    if (!auditSummary) return new Set<string>();
    const set = new Set<string>();
    auditSummary.forEach((r) => {
      if (r.totalTriggers > 0) set.add(r.alertId);
    });
    return set;
  }, [auditSummary]);

  const filteredBrowseAlerts = useMemo(() => {
    if (!allAlerts || !browseApplied) return [];
    let list = allAlerts;
    if (triggerStatus === "triggered") {
      list = list.filter((a) => triggeredAlertIdsInPeriod.has(a.alertId));
    } else if (triggerStatus === "not_triggered") {
      list = list.filter((a) => !triggeredAlertIdsInPeriod.has(a.alertId));
    }
    if (selectedExchanges.length) {
      list = list.filter((a) => a.exchange && selectedExchanges.includes(a.exchange));
    }
    if (selectedActions.length) {
      list = list.filter((a) => a.action && selectedActions.includes(a.action));
    }
    if (inAnyPortfolio) {
      list = list.filter((a) => a.ticker && portfolioTickers.set.has(a.ticker));
    }
    if (selectedPortfolioIds.length) {
      list = list.filter((a) => {
        if (!a.ticker) return false;
        return selectedPortfolioIds.some(
          (pid) => portfolioTickers.map[pid]?.includes(a.ticker)
        );
      });
    }
    if (selectedConditions.length) {
      list = list.filter((a) => {
        const conds = getAlertConditionStrings(a);
        return selectedConditions.some((c) => conds.includes(c));
      });
    }
    return list;
  }, [
    allAlerts,
    browseApplied,
    triggerStatus,
    triggeredAlertIdsInPeriod,
    selectedExchanges,
    selectedActions,
    inAnyPortfolio,
    selectedPortfolioIds,
    selectedConditions,
    portfolioTickers,
  ]);

  const triggerInfoByAlertId = useMemo(() => {
    const map: Record<string, { count: number; lastCheck: Date | undefined }> = {};
    auditSummary?.forEach((r) => {
      const cur = map[r.alertId] ?? { count: 0, lastCheck: undefined };
      cur.count += r.totalTriggers ?? 0;
      if (r.lastCheck) {
        const d = new Date(r.lastCheck);
        if (!cur.lastCheck || d > cur.lastCheck) cur.lastCheck = d;
      }
      map[r.alertId] = cur;
    });
    return map;
  }, [auditSummary]);

  const handleSearch = () => {
    setHasSearched(true);
    if (searchResults?.length === 1) setSelectedTicker(searchResults[0].ticker);
  };

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <SearchIcon className="size-6" />
          Alert History Lookup
        </h1>
        <p className="text-muted-foreground text-sm mt-1">
          Search for alerts by ticker or company name, or browse with filters.
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v)}>
        <TabsList>
          <TabsTrigger value="search">Search by Ticker/Company</TabsTrigger>
          <TabsTrigger value="browse">Browse with Filters</TabsTrigger>
        </TabsList>

        <TabsContent value="search" className="mt-4 space-y-4">
          <div className="flex gap-2 flex-wrap items-end">
            <div className="flex-1 min-w-[200px]">
              <label className="text-xs text-muted-foreground block mb-1">
                Search by ticker or company name
              </label>
              <Input
                placeholder="e.g. AAPL or Apple"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              />
            </div>
            <Button onClick={handleSearch} type="button">
              <SearchIcon className="size-4 mr-1" />
              Search
            </Button>
          </div>

          {!hasSearched || searchTerm.length < 2 ? (
            <>
              <p className="text-sm text-muted-foreground">
                Enter at least 2 characters and click Search to find companies.
              </p>
              <div className="flex flex-wrap gap-2">
                {POPULAR_TICKERS.map((t) => (
                  <Button
                    key={t}
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      setSearchTerm(t);
                      setSelectedTicker(t);
                      setHasSearched(true);
                    }}
                  >
                    {t}
                  </Button>
                ))}
              </div>
            </>
          ) : searchLoading ? (
            <p className="text-sm text-muted-foreground">Searching…</p>
          ) : !searchResults?.length ? (
            <p className="text-sm text-amber-600">No companies found matching &quot;{searchTerm}&quot;</p>
          ) : (
            <>
              {searchResults.length > 1 && (
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">Select company</label>
                  <Select
                    value={selectedTicker ?? ""}
                    onValueChange={(v) => setSelectedTicker(v)}
                  >
                    <SelectTrigger className="w-full max-w-md">
                      <SelectValue placeholder="Select…" />
                    </SelectTrigger>
                    <SelectContent>
                      {searchResults.map((r) => (
                        <SelectItem key={r.ticker} value={r.ticker}>
                          {r.ticker} — {r.name} ({r.exchange})
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}

              {selectedCompany && (
                <>
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">Company</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                        <div>
                          <p className="text-muted-foreground text-xs">Ticker</p>
                          <p className="font-semibold">{selectedCompany.ticker}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground text-xs">Company</p>
                          <p className="font-semibold truncate" title={selectedCompany.name}>
                            {selectedCompany.name.length > 30 ? selectedCompany.name.slice(0, 30) + "…" : selectedCompany.name}
                          </p>
                        </div>
                        <div>
                          <p className="text-muted-foreground text-xs">Exchange</p>
                          <p className="font-semibold">{selectedCompany.exchange}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground text-xs">Type</p>
                          <p className="font-semibold">{selectedCompany.type}</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">Active alerts ({alertsForTicker.length})</CardTitle>
                    </CardHeader>
                    <CardContent>
                      {alertsForTicker.length === 0 ? (
                        <p className="text-sm text-muted-foreground">
                          No active alerts for {selectedCompany.ticker}.
                        </p>
                      ) : (
                        <div className="space-y-2">
                          {alertsForTicker.map((alert) => (
                            <details key={alert.alertId} className="border rounded-md">
                              <summary className="px-3 py-2 cursor-pointer list-none font-medium">
                                {alert.name} — {alert.timeframe} | {alert.action}
                              </summary>
                              <div className="px-3 pb-3 text-sm border-t space-y-1">
                                <p>Timeframe: {alert.timeframe} · Action: {alert.action}</p>
                                <p>Exchange: {alert.exchange ?? "—"}</p>
                                {getAlertConditionStrings(alert).length > 0 && (
                                  <div>
                                    <p className="font-medium">Conditions:</p>
                                    {getAlertConditionStrings(alert).map((c, i) => (
                                      <pre key={i} className="text-xs bg-muted p-2 rounded mt-1">
                                        {c}
                                      </pre>
                                    ))}
                                  </div>
                                )}
                              </div>
                            </details>
                          ))}
                        </div>
                      )}
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">Alert trigger history</CardTitle>
                      <div className="flex flex-wrap gap-4 items-center mt-2">
                        <label className="flex items-center gap-2 text-sm">
                          <input
                            type="checkbox"
                            checked={showAllEvaluations}
                            onChange={(e) => setShowAllEvaluations(e.target.checked)}
                          />
                          Show all evaluations (not just triggers)
                        </label>
                        <div className="flex items-center gap-2">
                          <label className="text-sm">Days of history:</label>
                          <Input
                            type="number"
                            min={1}
                            max={365}
                            value={daysBack}
                            onChange={(e) => setDaysBack(Number(e.target.value) || 30)}
                            className="w-20"
                          />
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      {historyLoading ? (
                        <p className="text-sm text-muted-foreground">Loading history…</p>
                      ) : !historyRows?.length ? (
                        <p className="text-sm text-muted-foreground">
                          No trigger history found for {selectedCompany.ticker}.
                        </p>
                      ) : (
                        <TriggerHistoryList
                          rows={historyRows}
                          showAll={showAllEvaluations}
                          daysBack={daysBack}
                        />
                      )}
                    </CardContent>
                  </Card>
                </>
              )}
            </>
          )}
        </TabsContent>

        <TabsContent value="browse" className="mt-4 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Filters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">Days</label>
                  <Select value={String(browseDays)} onValueChange={(v) => setBrowseDays(Number(v))}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {[7, 14, 30, 60, 90].map((d) => (
                        <SelectItem key={d} value={String(d)}>{d} days</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">Trigger status</label>
                  <Select
                    value={triggerStatus}
                    onValueChange={(v: "all" | "triggered" | "not_triggered") => setTriggerStatus(v)}
                  >
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All</SelectItem>
                      <SelectItem value="triggered">Triggered only</SelectItem>
                      <SelectItem value="not_triggered">Not triggered</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex items-end">
                  <label className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={inAnyPortfolio}
                      onChange={(e) => setInAnyPortfolio(e.target.checked)}
                    />
                    In any portfolio
                  </label>
                </div>
              </div>
              {portfolios && portfolios.length > 0 && (
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">Specific portfolios</label>
                  <div className="flex flex-wrap gap-3">
                    {portfolios.map((p) => (
                      <label key={p.portfolioId} className="flex items-center gap-2 text-sm">
                        <input
                          type="checkbox"
                          checked={selectedPortfolioIds.includes(p.portfolioId)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedPortfolioIds((prev) => [...prev, p.portfolioId]);
                            } else {
                              setSelectedPortfolioIds((prev) => prev.filter((id) => id !== p.portfolioId));
                            }
                          }}
                        />
                        {p.name}
                      </label>
                    ))}
                  </div>
                </div>
              )}
              <div className="flex flex-wrap gap-2">
                <Button onClick={() => setBrowseApplied(true)}>Apply filters</Button>
              </div>
            </CardContent>
          </Card>

          {browseApplied && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Results: {filteredBrowseAlerts.length} alerts</CardTitle>
                <p className="text-muted-foreground text-xs">
                  Last {browseDays} days · {triggerStatus === "triggered" ? "Triggered only" : triggerStatus === "not_triggered" ? "Not triggered" : "All"}.
                </p>
              </CardHeader>
              <CardContent>
                {alertsLoading ? (
                  <p className="text-sm text-muted-foreground">Loading alerts…</p>
                ) : filteredBrowseAlerts.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No alerts match the filters.</p>
                ) : (
                  <div className="space-y-2">
                    {filteredBrowseAlerts.map((alert) => {
                      const info = triggerInfoByAlertId[alert.alertId];
                      const count = info?.count ?? 0;
                      const lastTriggered = info?.lastCheck;
                      const portfolioNames = alert.ticker
                        ? (portfolios ?? [])
                            .filter((p) => p.tickers?.includes(alert.ticker))
                            .map((p) => p.name)
                        : [];
                      return (
                        <details key={alert.alertId} className="border rounded-md">
                          <summary className="px-3 py-2 cursor-pointer list-none font-medium flex items-center gap-2">
                            <strong>{alert.ticker}</strong> — {alert.name}{" "}
                            <span className="text-muted-foreground font-normal">
                              ({count} trigger{count !== 1 ? "s" : ""} in range)
                            </span>
                            <ChevronDownIcon className="size-4 ml-auto" />
                          </summary>
                          <div className="px-3 pb-3 text-sm border-t space-y-2">
                            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                              <p>Stock: {alert.stockName ?? alert.ticker}</p>
                              <p>Exchange: {alert.exchange ?? "—"}</p>
                              <p>Portfolio: {portfolioNames.length ? portfolioNames.join(", ") : "None"}</p>
                            </div>
                            <p>Action: {alert.action ?? "—"} · Timeframe: {alert.timeframe ?? "—"}</p>
                            {lastTriggered && (
                              <p>Last triggered: {formatAuditDateTime(lastTriggered)}</p>
                            )}
                            {getAlertConditionStrings(alert).length > 0 && (
                              <div>
                                <p className="font-medium">Conditions:</p>
                                {getAlertConditionStrings(alert).map((c, i) => (
                                  <pre key={i} className="text-xs bg-muted p-2 rounded mt-1">{c}</pre>
                                ))}
                              </div>
                            )}
                            <Link
                              href={`/alerts/history?tab=search&ticker=${encodeURIComponent(alert.ticker ?? "")}`}
                            >
                              <Button variant="outline" size="sm">View full history</Button>
                            </Link>
                          </div>
                        </details>
                      );
                    })}
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>

      <p className="text-muted-foreground text-xs border-t pt-4">
        Data from alert audit logs and configured alerts.
      </p>
    </div>
  );
}

export default function AlertHistoryPage() {
  return (
    <Suspense fallback={<div className="p-6 text-muted-foreground">Loading…</div>}>
      <AlertHistoryContent />
    </Suspense>
  );
}
