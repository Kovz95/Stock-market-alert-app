import type { AlertData } from "@/actions/alert-actions";

/**
 * Extract unique condition display strings from an alert (for filters and display).
 * Handles conditions as array of { conditions: string } or struct.
 */
export function getAlertConditionStrings(alert: AlertData): string[] {
  const raw = alert.conditions;
  if (raw == null) return [];
  if (Array.isArray(raw)) {
    const strings: string[] = [];
    for (const item of raw) {
      if (item && typeof item === "object" && "conditions" in item) {
        const s = (item as { conditions?: string }).conditions;
        if (typeof s === "string" && s.trim()) strings.push(s.trim());
      }
    }
    return [...new Set(strings)];
  }
  if (typeof raw === "object" && raw !== null) {
    return Object.values(raw)
      .filter((v): v is string => typeof v === "string" && v.trim().length > 0)
      .map((s) => s.trim());
  }
  return [];
}
