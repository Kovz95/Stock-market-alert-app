-- name: GetStockEconomyBySymbol :one
SELECT symbol, rbics_economy, asset_type
FROM stock_metadata
WHERE symbol = $1;
