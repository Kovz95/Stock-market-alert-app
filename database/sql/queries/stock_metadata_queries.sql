-- name: GetStockEconomyBySymbol :one
SELECT symbol, rbics_economy, asset_type
FROM stock_metadata
WHERE symbol = $1;

-- Scheduler: load metadata for all tickers needed by alert formatting and Discord routing.

-- name: ListStockMetadataForAlerts :many
SELECT symbol, name, isin, exchange, country, rbics_economy, asset_type
FROM stock_metadata
ORDER BY symbol;

-- name: GetStockMetadataBySymbol :one
SELECT symbol, name, isin, exchange, country, rbics_economy, asset_type
FROM stock_metadata
WHERE symbol = $1;
