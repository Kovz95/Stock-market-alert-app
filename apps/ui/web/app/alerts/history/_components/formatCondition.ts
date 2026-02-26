/**
 * Format trigger reason / condition details for display (JSON or string to readable lines).
 */
export function formatCondition(details: string | Record<string, unknown> | null | undefined): string {
  if (details == null || details === "") return "N/A";
  if (typeof details === "string") {
    try {
      const parsed = JSON.parse(details) as unknown;
      if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
        return Object.entries(parsed)
          .map(([k, v]) => `• ${k}: ${String(v)}`)
          .join("\n");
      }
      return details;
    } catch {
      return details;
    }
  }
  if (typeof details === "object" && details !== null) {
    return Object.entries(details)
      .map(([k, v]) => `• ${k}: ${String(v)}`)
      .join("\n");
  }
  return String(details);
}
