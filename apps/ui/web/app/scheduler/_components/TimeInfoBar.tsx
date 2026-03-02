"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useEffect, useState } from "react";

function getEasternTime(): Date {
  return new Date(
    new Date().toLocaleString("en-US", { timeZone: "America/New_York" })
  );
}

function isDST(): boolean {
  const jan = new Date(new Date().getFullYear(), 0, 1).getTimezoneOffset();
  const jul = new Date(new Date().getFullYear(), 6, 1).getTimezoneOffset();
  const et = new Date(
    new Date().toLocaleString("en-US", { timeZone: "America/New_York" })
  );
  // If current offset matches the smaller offset (summer), it's DST
  const etOffset = new Date().getTimezoneOffset();
  // Simpler: check timezone abbreviation
  const tzStr = new Date().toLocaleString("en-US", {
    timeZone: "America/New_York",
    timeZoneName: "short",
  });
  return tzStr.includes("EDT");
}

const DAYS = [
  "Sunday",
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
];

export function TimeInfoBar() {
  const [now, setNow] = useState<Date | null>(null);

  useEffect(() => {
    setNow(getEasternTime());
    const interval = setInterval(() => setNow(getEasternTime()), 1000);
    return () => clearInterval(interval);
  }, []);

  const dst = now ? isDST() : null;

  return (
    <div className="grid gap-4 md:grid-cols-3">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Current Time (ET)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <span className="text-xl font-semibold tabular-nums">
            {now
              ? now.toLocaleTimeString("en-US", {
                  hour: "2-digit",
                  minute: "2-digit",
                  second: "2-digit",
                  hour12: true,
                })
              : "--:--:-- --"}
          </span>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Day of Week
          </CardTitle>
        </CardHeader>
        <CardContent>
          <span className="text-xl font-semibold">
            {now ? DAYS[now.getDay()] : "---"}
          </span>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            DST Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <span className="text-xl font-semibold">
            {dst === null ? "---" : dst ? "EDT (Daylight)" : "EST (Standard)"}
          </span>
        </CardContent>
      </Card>
    </div>
  );
}
