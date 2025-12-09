-- PostgreSQL schema for Stock Alert App
-- -------------------------------------
-- This file mirrors the SQLite structures currently used by the app and
-- introduces the indexes/constraints we rely on for performance.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS daily_prices (
    ticker          TEXT        NOT NULL,
    date            DATE        NOT NULL,
    open            DOUBLE PRECISION,
    high            DOUBLE PRECISION,
    low             DOUBLE PRECISION,
    close           DOUBLE PRECISION NOT NULL,
    volume          BIGINT,
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_daily_prices_date
    ON daily_prices (date DESC);


CREATE TABLE IF NOT EXISTS hourly_prices (
    ticker          TEXT        NOT NULL,
    datetime        TIMESTAMPTZ NOT NULL,
    open            DOUBLE PRECISION,
    high            DOUBLE PRECISION,
    low             DOUBLE PRECISION,
    close           DOUBLE PRECISION NOT NULL,
    volume          BIGINT,
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (ticker, datetime)
);

CREATE INDEX IF NOT EXISTS idx_hourly_prices_ticker_datetime
    ON hourly_prices (ticker, datetime DESC);


CREATE TABLE IF NOT EXISTS weekly_prices (
    ticker          TEXT        NOT NULL,
    week_ending     DATE        NOT NULL,
    open            DOUBLE PRECISION,
    high            DOUBLE PRECISION,
    low             DOUBLE PRECISION,
    close           DOUBLE PRECISION NOT NULL,
    volume          BIGINT,
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (ticker, week_ending)
);

CREATE INDEX IF NOT EXISTS idx_weekly_prices_ticker_week
    ON weekly_prices (ticker, week_ending DESC);


CREATE TABLE IF NOT EXISTS ticker_metadata (
    ticker          TEXT PRIMARY KEY,
    first_date      DATE,
    last_date       DATE,
    total_records   INTEGER,
    last_update     TIMESTAMPTZ,
    exchange        TEXT,
    asset_type      TEXT
);


CREATE TABLE IF NOT EXISTS daily_move_stats (
    ticker          TEXT        NOT NULL,
    date            DATE        NOT NULL,
    pct_change      DOUBLE PRECISION,
    mean_change     DOUBLE PRECISION,
    std_change      DOUBLE PRECISION,
    zscore          DOUBLE PRECISION,
    sigma_level     INTEGER,
    direction       TEXT,
    magnitude       TEXT,
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (ticker, date)
);


CREATE TABLE IF NOT EXISTS alert_audits (
    id                      BIGSERIAL PRIMARY KEY,
    timestamp               TIMESTAMPTZ NOT NULL,
    alert_id                TEXT NOT NULL,
    ticker                  TEXT NOT NULL,
    stock_name              TEXT,
    exchange                TEXT,
    timeframe               TEXT,
    action                  TEXT,
    evaluation_type         TEXT NOT NULL,
    price_data_pulled       BOOLEAN,
    price_data_source       TEXT,
    conditions_evaluated    BOOLEAN,
    alert_triggered         BOOLEAN,
    trigger_reason          TEXT,
    execution_time_ms       INTEGER,
    cache_hit               BOOLEAN,
    error_message           TEXT,
    additional_data         JSONB
);

CREATE INDEX IF NOT EXISTS idx_alert_audits_ticker_ts
    ON alert_audits (ticker, timestamp DESC);


CREATE TABLE IF NOT EXISTS continuous_prices (
    symbol              TEXT        NOT NULL,
    date                DATE        NOT NULL,
    open                DOUBLE PRECISION,
    high                DOUBLE PRECISION,
    low                 DOUBLE PRECISION,
    close               DOUBLE PRECISION,
    volume              BIGINT,
    adjustment_method   TEXT DEFAULT 'panama',
    front_month         TEXT,
    roll_date           DATE,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, date)
);

CREATE INDEX IF NOT EXISTS idx_continuous_prices_symbol_date
    ON continuous_prices (symbol, date DESC);


CREATE TABLE IF NOT EXISTS futures_metadata (
    symbol                  TEXT PRIMARY KEY,
    name                    TEXT,
    exchange                TEXT,
    category                TEXT,
    multiplier              DOUBLE PRECISION,
    min_tick                DOUBLE PRECISION,
    currency                TEXT,
    contract_id             INTEGER,
    last_update             TIMESTAMPTZ,
    front_month             TEXT,
    next_roll_date          DATE,
    data_quality_score      DOUBLE PRECISION DEFAULT 1.0
);

-- ---------------------------------------------------------------------------
-- Equity metadata previously sourced from JSON
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS stock_metadata (
    symbol                TEXT PRIMARY KEY,
    isin                  TEXT,
    name                  TEXT,
    exchange              TEXT,
    country               TEXT,
    rbics_economy         TEXT,
    rbics_sector          TEXT,
    rbics_subsector       TEXT,
    rbics_industry_group  TEXT,
    rbics_industry        TEXT,
    rbics_subindustry     TEXT,
    closing_price         DOUBLE PRECISION,
    market_value          DOUBLE PRECISION,
    sales                 DOUBLE PRECISION,
    avg_daily_volume      DOUBLE PRECISION,
    data_source           TEXT,
    last_updated          TIMESTAMPTZ,
    asset_type            TEXT,
    raw_payload           JSONB
);

CREATE INDEX IF NOT EXISTS idx_stock_metadata_exchange
    ON stock_metadata (exchange);

CREATE INDEX IF NOT EXISTS idx_stock_metadata_country
    ON stock_metadata (country);

-- ---------------------------------------------------------------------------
-- Alerts previously stored in alerts.json
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS alerts (
    alert_id           UUID PRIMARY KEY,
    name               TEXT NOT NULL,
    stock_name         TEXT,
    ticker             TEXT,
    ticker1            TEXT,
    ticker2            TEXT,
    conditions         JSONB,
    combination_logic  TEXT,
    last_triggered     TIMESTAMPTZ,
    action             TEXT,
    timeframe          TEXT,
    exchange           TEXT,
    country            TEXT,
    ratio              TEXT,
    is_ratio           BOOLEAN DEFAULT FALSE,
    adjustment_method  TEXT,
    dtp_params         JSONB,
    multi_timeframe_params JSONB,
    mixed_timeframe_params JSONB,
    raw_payload        JSONB,
    created_at         TIMESTAMPTZ DEFAULT NOW(),
    updated_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_ticker
    ON alerts (ticker);

CREATE INDEX IF NOT EXISTS idx_alerts_exchange
    ON alerts (exchange);

CREATE INDEX IF NOT EXISTS idx_alerts_ratio
    ON alerts (is_ratio, ratio);

-- ---------------------------------------------------------------------------
-- Portfolios previously stored in portfolios.json
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS portfolios (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    discord_webhook  TEXT,
    enabled          BOOLEAN DEFAULT TRUE,
    created_date     TIMESTAMPTZ,
    last_updated     TIMESTAMPTZ,
    raw_payload      JSONB
);

CREATE TABLE IF NOT EXISTS portfolio_stocks (
    portfolio_id TEXT REFERENCES portfolios(id) ON DELETE CASCADE,
    ticker       TEXT NOT NULL,
    PRIMARY KEY (portfolio_id, ticker)
);

CREATE INDEX IF NOT EXISTS idx_portfolio_stocks_symbol
    ON portfolio_stocks (ticker);

-- ---------------------------------------------------------------------------
-- Generic JSON document storage (replacement for legacy JSON files)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS app_documents (
    document_key   TEXT PRIMARY KEY,
    payload        JSONB NOT NULL,
    source_path    TEXT,
    updated_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_app_documents_updated_at
    ON app_documents (updated_at DESC);
