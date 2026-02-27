FROM golang:1.24-alpine AS builder

RUN apk add --no-cache git

WORKDIR /build

# Copy go.work and all module files first for caching
COPY gen/go ./gen/go
COPY database ./database
COPY apps/grpc/ ./apps/grpc/
COPY calendar ./calendar
COPY discord ./discord
COPY expr ./expr
COPY indicator ./indicator
COPY apps/scheduler apps/scheduler

# Download dependencies
WORKDIR /build/apps/grpc/alert_service
RUN go mod download

# Copy all source code
WORKDIR /build
COPY database/ database/
COPY gen/go/ gen/go/
COPY apps/grpc/alert_service/ apps/grpc/alert_service/

# Build the binary
WORKDIR /build/apps/grpc/alert_service
RUN CGO_ENABLED=0 GOOS=linux go build -o /alert_service .

# --- Runtime stage ---
FROM alpine:3.21

RUN apk add --no-cache ca-certificates

# Install grpc_health_probe
RUN GRPC_HEALTH_PROBE_VERSION=v0.4.35 && \
    wget -qO/bin/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64 && \
    chmod +x /bin/grpc_health_probe

COPY --from=builder /alert_service /alert_service

EXPOSE 50051

ENTRYPOINT ["/alert_service"]
