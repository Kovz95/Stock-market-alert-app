-- Price database read path: stats, metadata, and list prices.

-- name: GetDailyStats :one
SELECT
    COUNT(*)::bigint AS record_count,
    COUNT(DISTINCT ticker)::int AS ticker_count,
    MIN(date)::date AS min_date,
    MAX(date)::date AS max_date
FROM daily_prices;

-- name: GetHourlyStats :one
SELECT
    COUNT(*)::bigint AS record_count,
    COUNT(DISTINCT ticker)::int AS ticker_count,
    MIN(datetime) AS min_datetime,
    MAX(datetime) AS max_datetime
FROM hourly_prices;

-- name: GetWeeklyStats :one
SELECT
    COUNT(*)::bigint AS record_count,
    COUNT(DISTINCT ticker)::int AS ticker_count,
    MIN(week_ending)::date AS min_date,
    MAX(week_ending)::date AS max_date
FROM weekly_prices;

-- name: ListStockMetadataForPriceDb :many
SELECT symbol, name, exchange, isin
FROM stock_metadata
ORDER BY symbol;

-- ListDailyPrices: tickers null or empty = no ticker filter. day_filter: 0 = all, 1 = weekdays (Mon-Fri), 2 = weekends (Sat-Sun).
-- name: ListDailyPrices :many
SELECT ticker, date, open, high, low, close, volume
FROM daily_prices
WHERE (COALESCE(array_length(sqlc.arg(tickers)::text[], 1), 0) = 0 OR ticker = ANY(sqlc.arg(tickers)))
  AND (sqlc.arg(start_date)::date IS NULL OR date >= sqlc.arg(start_date))
  AND (sqlc.arg(end_date)::date IS NULL OR date <= sqlc.arg(end_date))
  AND (
    sqlc.arg(day_filter)::int IS NULL OR sqlc.arg(day_filter) = 0
    OR (sqlc.arg(day_filter) = 1 AND EXTRACT(ISODOW FROM date) BETWEEN 1 AND 5)
    OR (sqlc.arg(day_filter) = 2 AND EXTRACT(ISODOW FROM date) IN (6, 7))
  )
ORDER BY ticker, date DESC
LIMIT sqlc.arg(limit_rows);

-- name: ListHourlyPrices :many
SELECT ticker, datetime, open, high, low, close, volume
FROM hourly_prices
WHERE (COALESCE(array_length(sqlc.arg(tickers)::text[], 1), 0) = 0 OR ticker = ANY(sqlc.arg(tickers)))
  AND (sqlc.arg(start_ts)::timestamptz IS NULL OR datetime >= sqlc.arg(start_ts))
  AND (sqlc.arg(end_ts)::timestamptz IS NULL OR datetime <= sqlc.arg(end_ts))
  AND (
    sqlc.arg(day_filter)::int IS NULL OR sqlc.arg(day_filter) = 0
    OR (sqlc.arg(day_filter) = 1 AND EXTRACT(ISODOW FROM datetime) BETWEEN 1 AND 5)
    OR (sqlc.arg(day_filter) = 2 AND EXTRACT(ISODOW FROM datetime) IN (6, 7))
  )
ORDER BY ticker, datetime DESC
LIMIT sqlc.arg(limit_rows);

-- name: ListWeeklyPrices :many
SELECT ticker, week_ending, open, high, low, close, volume
FROM weekly_prices
WHERE (COALESCE(array_length(sqlc.arg(tickers)::text[], 1), 0) = 0 OR ticker = ANY(sqlc.arg(tickers)))
  AND (sqlc.arg(start_date)::date IS NULL OR week_ending >= sqlc.arg(start_date))
  AND (sqlc.arg(end_date)::date IS NULL OR week_ending <= sqlc.arg(end_date))
  AND (
    sqlc.arg(day_filter)::int IS NULL OR sqlc.arg(day_filter) = 0
    OR (sqlc.arg(day_filter) = 1 AND EXTRACT(ISODOW FROM week_ending) BETWEEN 1 AND 5)
    OR (sqlc.arg(day_filter) = 2 AND EXTRACT(ISODOW FROM week_ending) IN (6, 7))
  )
ORDER BY ticker, week_ending DESC
LIMIT sqlc.arg(limit_rows);

-- Stale scan: last date per ticker for daily (join metadata for exchange/name; Go filters by expected date).
-- name: LastDailyPerTicker :many
SELECT d.ticker, d.last_date, m.name, m.exchange
FROM (
    SELECT ticker, MAX(date)::date AS last_date
    FROM daily_prices
    GROUP BY ticker
) d
LEFT JOIN stock_metadata m ON m.symbol = d.ticker
ORDER BY d.last_date ASC;

-- Stale weekly: tickers where max(week_ending) < expected_week_ending; optional limit.
-- name: StaleWeeklyTickers :many
SELECT w.ticker, w.last_date, (sqlc.arg(expected_week_ending)::date - w.last_date)::int AS days_old, m.name, m.exchange
FROM (
    SELECT ticker, MAX(week_ending)::date AS last_date
    FROM weekly_prices
    GROUP BY ticker
    HAVING MAX(week_ending) < sqlc.arg(expected_week_ending)::date
) w
LEFT JOIN stock_metadata m ON m.symbol = w.ticker
ORDER BY w.last_date ASC
LIMIT sqlc.arg(limit_rows);

-- Stale hourly: last datetime per ticker (Go computes expected hour and hours_behind).
-- name: LastHourlyPerTicker :many
SELECT ticker, MAX(datetime) AS last_dt
FROM hourly_prices
GROUP BY ticker
ORDER BY MAX(datetime) ASC;

-- Hourly data quality: stale count (last_dt < now - 48h).
-- name: HourlyQualityStale :one
WITH last_points AS (
    SELECT ticker, MAX(datetime) AS last_dt
    FROM hourly_prices
    GROUP BY ticker
)
SELECT
    COUNT(*)::int AS total_tickers,
    COUNT(*) FILTER (WHERE last_dt < NOW() - INTERVAL '48 hours')::int AS stale_tickers,
    MIN(last_dt) FILTER (WHERE last_dt < NOW() - INTERVAL '48 hours') AS oldest_stale
FROM last_points;

-- Hourly data quality: gap metrics (trading-hour gaps in last 60 days).
-- name: HourlyQualityGaps :one
WITH recent AS (
    SELECT ticker, datetime
    FROM hourly_prices
    WHERE datetime >= NOW() - INTERVAL '60 days'
),
ordered AS (
    SELECT ticker, datetime,
           LEAD(datetime) OVER (PARTITION BY ticker ORDER BY datetime) AS next_dt
    FROM recent
),
gaps AS (
    SELECT ticker,
           EXTRACT(EPOCH FROM (next_dt - datetime))/3600.0 AS gap_hours,
           (SELECT COUNT(*)::float FROM generate_series(datetime, next_dt - INTERVAL '1 hour', INTERVAL '1 hour') gs(dt)
            WHERE EXTRACT(ISODOW FROM gs.dt) BETWEEN 1 AND 5) AS trading_gap_hours
    FROM ordered
    WHERE next_dt IS NOT NULL
)
SELECT
    COUNT(DISTINCT ticker) FILTER (WHERE trading_gap_hours > 72)::int AS gap_tickers,
    COALESCE(MAX(trading_gap_hours), 0)::double precision AS worst_gap_hours,
    COALESCE(MAX(gap_hours), 0)::double precision AS worst_calendar_gap_hours
FROM gaps;
