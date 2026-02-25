-- name: ListAlerts :many
SELECT
    alert_id, name, stock_name, ticker, ticker1, ticker2,
    conditions, combination_logic, last_triggered, action,
    timeframe, exchange, country, ratio, is_ratio,
    adjustment_method, dtp_params, multi_timeframe_params,
    mixed_timeframe_params, raw_payload, created_at, updated_at
FROM alerts
ORDER BY updated_at DESC, name ASC;

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

-- name: BulkUpdateLastTriggered :exec
UPDATE alerts SET
    last_triggered = $2,
    updated_at = NOW()
WHERE alert_id = $1;
