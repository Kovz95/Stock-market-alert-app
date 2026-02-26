/**
 * Client-safe enum values for the Price Database UI.
 * Matches proto/price/v1/price.proto (Timeframe and DayFilter).
 */

export const Timeframe = {
  UNSPECIFIED: 0,
  HOURLY: 1,
  DAILY: 2,
  WEEKLY: 3,
} as const;

export type TimeframeValue = (typeof Timeframe)[keyof typeof Timeframe];

export const DayFilter = {
  UNSPECIFIED: 0,
  ALL: 1,
  WEEKDAYS: 2,
  WEEKENDS: 3,
} as const;

export type DayFilterValue = (typeof DayFilter)[keyof typeof DayFilter];
