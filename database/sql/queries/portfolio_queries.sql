-- name: ListPortfoliosForFanout :many
SELECT id, name, discord_webhook, enabled
FROM portfolios
WHERE enabled = TRUE
  AND discord_webhook IS NOT NULL
  AND discord_webhook <> '';

-- name: ListPortfolioStocks :many
SELECT portfolio_id, ticker
FROM portfolio_stocks;
