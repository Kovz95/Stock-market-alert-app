# Build scheduler and worker binaries. Context must be repo root.
# Do not copy go.work so Go does not try to load other workspace modules (grpc, gen/go).
FROM golang:1.24-alpine AS builder

RUN apk add --no-cache git

WORKDIR /build

# Copy only modules referenced by scheduler go.mod replace directives
COPY alert ./alert
COPY calendar ./calendar
COPY database ./database
COPY discord ./discord
COPY expr ./expr
COPY indicator ./indicator
COPY apps/scheduler ./apps/scheduler

# Download scheduler deps (uses apps/scheduler/go.mod and replace paths only)
WORKDIR /build/apps/scheduler
RUN go mod download

# Copy full source (in case any local replace targets changed)
WORKDIR /build
COPY alert/ alert/
COPY calendar/ calendar/
COPY database/ database/
COPY discord/ discord/
COPY expr/ expr/
COPY indicator/ indicator/
COPY apps/scheduler/ apps/scheduler/

# Build worker binary
WORKDIR /build/apps/scheduler
RUN CGO_ENABLED=0 GOOS=linux go build -o /worker ./cmd/worker

# --- Runtime stage ---
FROM alpine:3.21

RUN apk add --no-cache ca-certificates

COPY --from=builder /worker /worker

ENTRYPOINT ["/worker"]
