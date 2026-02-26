# Price Service (gRPC)

Read-only backend for the Price Database UI: stock metadata, database stats, and OHLCV price data (daily, hourly, weekly).

## Run locally

```bash
export DATABASE_URL="postgres://user:pass@localhost:5432/stockalert"
export PORT=50051   # optional; default 50051
go run .
```

## Environment

| Variable        | Required | Description                          |
|----------------|----------|--------------------------------------|
| `DATABASE_URL` | Yes      | PostgreSQL connection string         |
| `PORT`         | No       | gRPC listen port (default: 50051)    |

## Docker

Build from the repo root (all gRPC services listen on 50051):

```bash
docker build -f apps/grpc/price_service/Dockerfile -t price_service .
docker run --rm -e DATABASE_URL="postgres://..." -p 50051:50051 price_service
```

## Envoy

Envoy routes `/stockalert.price.v1.PriceService/` to the `price_service` cluster (hostname `price_service`, port 50051). Use `GRPC_ENDPOINT=localhost:8080` so the Next.js app talks to Envoy; Envoy forwards to each backend by service name.

## RPCs (Phase 1)

- `GetStockMetadataMap` – list all stock metadata (symbol, name, exchange, isin) for filters and ticker list.
- `GetFullStockMetadata` – list full stock metadata (all table columns plus ETF fields from `raw_payload`) for the Stock Database UI.
- `GetDatabaseStats` – record counts, ticker counts, and date ranges for hourly/daily/weekly tables.
- `LoadPriceData` – filtered OHLCV rows by timeframe, tickers, date range, day filter, and limit.
