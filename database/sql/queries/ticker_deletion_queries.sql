-- name: TickerExists :one
SELECT EXISTS(
    SELECT 1 FROM stock_metadata sm WHERE sm.symbol = $1
    UNION ALL
    SELECT 1 FROM ticker_metadata tm WHERE tm.ticker = $1
) AS exists;

-- name: CountTickerReferences :one
SELECT
    (SELECT COUNT(*) FROM stock_metadata sm    WHERE sm.symbol  = $1)::bigint AS stock_metadata,
    (SELECT COUNT(*) FROM ticker_metadata tm   WHERE tm.ticker  = $1)::bigint AS ticker_metadata,
    (SELECT COUNT(*) FROM daily_prices dp      WHERE dp.ticker  = $1)::bigint AS daily_prices,
    (SELECT COUNT(*) FROM hourly_prices hp     WHERE hp.ticker  = $1)::bigint AS hourly_prices,
    (SELECT COUNT(*) FROM weekly_prices wp     WHERE wp.ticker  = $1)::bigint AS weekly_prices,
    (SELECT COUNT(*) FROM continuous_prices cp WHERE cp.symbol  = $1)::bigint AS continuous_prices,
    (SELECT COUNT(*) FROM daily_move_stats dms WHERE dms.ticker = $1)::bigint AS daily_move_stats,
    (SELECT COUNT(*) FROM futures_metadata fm  WHERE fm.symbol  = $1)::bigint AS futures_metadata,
    (SELECT COUNT(*) FROM alerts ad            WHERE ad.ticker  = $1)::bigint AS alerts_direct,
    (SELECT COUNT(*) FROM alerts ar            WHERE ar.ticker1 = $1 OR ar.ticker2 = $1)::bigint AS alerts_ratio,
    (SELECT COUNT(*) FROM alert_audits aa      WHERE aa.ticker  = $1)::bigint AS alert_audits,
    (SELECT COUNT(*) FROM portfolio_stocks ps  WHERE ps.ticker  = $1)::bigint AS portfolio_stocks;

-- name: DeleteTickerFromPortfolioStocks :execrows
DELETE FROM portfolio_stocks WHERE ticker = $1;

-- name: DeleteTickerFromAlertAudits :execrows
DELETE FROM alert_audits WHERE ticker = $1;

-- name: DeleteTickerFromAlertsDirect :execrows
DELETE FROM alerts WHERE ticker = $1;

-- name: DeleteTickerFromAlertsRatio :execrows
DELETE FROM alerts WHERE ticker1 = $1 OR ticker2 = $1;

-- name: DeleteTickerFromDailyMoveStats :execrows
DELETE FROM daily_move_stats WHERE ticker = $1;

-- name: DeleteTickerFromDailyPrices :execrows
DELETE FROM daily_prices WHERE ticker = $1;

-- name: DeleteTickerFromHourlyPrices :execrows
DELETE FROM hourly_prices WHERE ticker = $1;

-- name: DeleteTickerFromWeeklyPrices :execrows
DELETE FROM weekly_prices WHERE ticker = $1;

-- name: DeleteTickerFromContinuousPrices :execrows
DELETE FROM continuous_prices WHERE symbol = $1;

-- name: DeleteTickerFromFuturesMetadata :execrows
DELETE FROM futures_metadata WHERE symbol = $1;

-- name: DeleteTickerFromTickerMetadata :execrows
DELETE FROM ticker_metadata WHERE ticker = $1;

-- name: DeleteTickerFromStockMetadata :execrows
DELETE FROM stock_metadata WHERE symbol = $1;
