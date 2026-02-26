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
