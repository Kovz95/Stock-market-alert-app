-- name: ListAlerts :many
SELECT
    alert_id, name, stock_name, ticker, ticker1, ticker2,
    conditions, combination_logic, last_triggered, action,
    timeframe, exchange, country, ratio, is_ratio,
    adjustment_method, dtp_params, multi_timeframe_params,
    mixed_timeframe_params, raw_payload, created_at, updated_at
FROM alerts
ORDER BY updated_at DESC, name ASC;

-- name: ListAlertsPaginated :many
SELECT
    alert_id, name, stock_name, ticker, ticker1, ticker2,
    conditions, combination_logic, last_triggered, action,
    timeframe, exchange, country, ratio, is_ratio,
    adjustment_method, dtp_params, multi_timeframe_params,
    mixed_timeframe_params, raw_payload, created_at, updated_at
FROM alerts
ORDER BY updated_at DESC, name ASC
LIMIT $1 OFFSET $2;

-- name: CountAlerts :one
SELECT COUNT(*) FROM alerts;

-- name: GetAlert :one
SELECT
    alert_id, name, stock_name, ticker, ticker1, ticker2,
    conditions, combination_logic, last_triggered, action,
    timeframe, exchange, country, ratio, is_ratio,
    adjustment_method, dtp_params, multi_timeframe_params,
    mixed_timeframe_params, raw_payload, created_at, updated_at
FROM alerts
WHERE alert_id = $1;

-- name: CreateAlert :one
INSERT INTO alerts (
    alert_id, name, stock_name, ticker, ticker1, ticker2,
    conditions, combination_logic, last_triggered, action,
    timeframe, exchange, country, ratio, is_ratio,
    adjustment_method, dtp_params, multi_timeframe_params,
    mixed_timeframe_params, raw_payload, created_at, updated_at
) VALUES (
    $1, $2, $3, $4, $5, $6,
    $7, $8, $9, $10,
    $11, $12, $13, $14, $15,
    $16, $17, $18,
    $19, $20, NOW(), NOW()
)
RETURNING *;

-- name: UpdateAlert :one
UPDATE alerts SET
    name = $2,
    stock_name = $3,
    ticker = $4,
    ticker1 = $5,
    ticker2 = $6,
    conditions = $7,
    combination_logic = $8,
    action = $9,
    timeframe = $10,
    exchange = $11,
    country = $12,
    ratio = $13,
    is_ratio = $14,
    adjustment_method = $15,
    dtp_params = $16,
    multi_timeframe_params = $17,
    mixed_timeframe_params = $18,
    raw_payload = $19,
    updated_at = NOW()
WHERE alert_id = $1
RETURNING *;

-- name: DeleteAlert :exec
DELETE FROM alerts WHERE alert_id = $1;

-- name: BulkDeleteAlerts :execrows
DELETE FROM alerts WHERE alert_id = ANY($1::uuid[]);

-- name: BulkUpdateLastTriggered :exec
UPDATE alerts SET
    last_triggered = $2,
    updated_at = NOW()
WHERE alert_id = $1;

-- Server-side filtered + paginated alert search (delete alerts page).

-- name: SearchAlertsPaginated :many
SELECT
    alert_id, name, stock_name, ticker, ticker1, ticker2,
    conditions, combination_logic, last_triggered, action,
    timeframe, exchange, country, ratio, is_ratio,
    adjustment_method, dtp_params, multi_timeframe_params,
    mixed_timeframe_params, raw_payload, created_at, updated_at
FROM alerts
WHERE
  (sqlc.arg(search)::text = '' OR
    name ILIKE '%' || sqlc.arg(search)::text || '%' OR
    ticker ILIKE '%' || sqlc.arg(search)::text || '%' OR
    stock_name ILIKE '%' || sqlc.arg(search)::text || '%')
  AND (cardinality(COALESCE(sqlc.arg(filter_exchanges)::text[], '{}'::text[])) = 0 OR exchange = ANY(COALESCE(sqlc.arg(filter_exchanges)::text[], '{}'::text[])))
  AND (cardinality(COALESCE(sqlc.arg(filter_timeframes)::text[], '{}'::text[])) = 0 OR timeframe = ANY(COALESCE(sqlc.arg(filter_timeframes)::text[], '{}'::text[])))
  AND (cardinality(COALESCE(sqlc.arg(filter_countries)::text[], '{}'::text[])) = 0 OR country = ANY(COALESCE(sqlc.arg(filter_countries)::text[], '{}'::text[])))
  AND (CASE sqlc.arg(triggered_filter)::text
    WHEN '' THEN TRUE
    WHEN 'never' THEN last_triggered IS NULL
    WHEN 'today' THEN last_triggered::date = CURRENT_DATE
    WHEN 'this_week' THEN last_triggered >= CURRENT_DATE - INTERVAL '7 days'
    WHEN 'this_month' THEN date_trunc('month', last_triggered) = date_trunc('month', CURRENT_DATE)
    WHEN 'this_year' THEN date_trunc('year', last_triggered) = date_trunc('year', CURRENT_DATE)
    ELSE TRUE END)
  AND (sqlc.arg(condition_search)::text = '' OR
    conditions::text ILIKE '%' || sqlc.arg(condition_search)::text || '%')
ORDER BY updated_at DESC, name ASC
LIMIT sqlc.arg(lim) OFFSET sqlc.arg(off);

-- name: CountSearchAlerts :one
SELECT COUNT(*) FROM alerts
WHERE
  (sqlc.arg(search)::text = '' OR
    name ILIKE '%' || sqlc.arg(search)::text || '%' OR
    ticker ILIKE '%' || sqlc.arg(search)::text || '%' OR
    stock_name ILIKE '%' || sqlc.arg(search)::text || '%')
  AND (cardinality(COALESCE(sqlc.arg(filter_exchanges)::text[], '{}'::text[])) = 0 OR exchange = ANY(COALESCE(sqlc.arg(filter_exchanges)::text[], '{}'::text[])))
  AND (cardinality(COALESCE(sqlc.arg(filter_timeframes)::text[], '{}'::text[])) = 0 OR timeframe = ANY(COALESCE(sqlc.arg(filter_timeframes)::text[], '{}'::text[])))
  AND (cardinality(COALESCE(sqlc.arg(filter_countries)::text[], '{}'::text[])) = 0 OR country = ANY(COALESCE(sqlc.arg(filter_countries)::text[], '{}'::text[])))
  AND (CASE sqlc.arg(triggered_filter)::text
    WHEN '' THEN TRUE
    WHEN 'never' THEN last_triggered IS NULL
    WHEN 'today' THEN last_triggered::date = CURRENT_DATE
    WHEN 'this_week' THEN last_triggered >= CURRENT_DATE - INTERVAL '7 days'
    WHEN 'this_month' THEN date_trunc('month', last_triggered) = date_trunc('month', CURRENT_DATE)
    WHEN 'this_year' THEN date_trunc('year', last_triggered) = date_trunc('year', CURRENT_DATE)
    ELSE TRUE END)
  AND (sqlc.arg(condition_search)::text = '' OR
    conditions::text ILIKE '%' || sqlc.arg(condition_search)::text || '%');

-- Scheduler queries: filter alerts by exchange(s) for a specific job run.

-- name: ListAlertsByExchange :many
SELECT
    alert_id, name, stock_name, ticker, ticker1, ticker2,
    conditions, combination_logic, last_triggered, action,
    timeframe, exchange, country, ratio, is_ratio,
    adjustment_method, dtp_params, multi_timeframe_params,
    mixed_timeframe_params, raw_payload, created_at, updated_at
FROM alerts
WHERE exchange = ANY(sqlc.arg(exchanges)::text[])
ORDER BY updated_at DESC, name ASC;

-- name: ListAlertsByExchangeAndTimeframe :many
SELECT
    alert_id, name, stock_name, ticker, ticker1, ticker2,
    conditions, combination_logic, last_triggered, action,
    timeframe, exchange, country, ratio, is_ratio,
    adjustment_method, dtp_params, multi_timeframe_params,
    mixed_timeframe_params, raw_payload, created_at, updated_at
FROM alerts
WHERE exchange = ANY(sqlc.arg(exchanges)::text[])
  AND timeframe = sqlc.arg(timeframe)
ORDER BY updated_at DESC, name ASC;

-- name: ListAlertsByExchangeAndTimeframes :many
SELECT
    alert_id, name, stock_name, ticker, ticker1, ticker2,
    conditions, combination_logic, last_triggered, action,
    timeframe, exchange, country, ratio, is_ratio,
    adjustment_method, dtp_params, multi_timeframe_params,
    mixed_timeframe_params, raw_payload, created_at, updated_at
FROM alerts
WHERE exchange = ANY(sqlc.arg(exchanges)::text[])
  AND timeframe = ANY(sqlc.arg(timeframes)::text[])
ORDER BY updated_at DESC, name ASC;

-- Audit trail: single insert for deferred audit records.

-- name: InsertAlertAudit :exec
INSERT INTO alert_audits (
    timestamp, alert_id, ticker, stock_name, exchange, timeframe, action,
    evaluation_type, price_data_pulled, price_data_source, conditions_evaluated,
    alert_triggered, trigger_reason, execution_time_ms, cache_hit,
    error_message, additional_data
) VALUES (
    $1, $2, $3, $4, $5, $6, $7,
    $8, $9, $10, $11,
    $12, $13, $14, $15,
    $16, $17
);

-- Bulk audit insert via COPY protocol (pgx CopyFrom).

-- name: CopyAlertAudits :copyfrom
INSERT INTO alert_audits (
    timestamp, alert_id, ticker, stock_name, exchange, timeframe, action,
    evaluation_type, price_data_pulled, price_data_source, conditions_evaluated,
    alert_triggered, trigger_reason, execution_time_ms, cache_hit,
    error_message, additional_data
) VALUES (
    $1, $2, $3, $4, $5, $6, $7,
    $8, $9, $10, $11,
    $12, $13, $14, $15,
    $16, $17
);
