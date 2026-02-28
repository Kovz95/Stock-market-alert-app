FROM golang:1.24-alpine AS builder

RUN apk add --no-cache git

WORKDIR /build

COPY gen/go ./gen/go
COPY database ./database
COPY calendar ./calendar
COPY apps/grpc/scheduler_service ./apps/grpc/scheduler_service

WORKDIR /build/apps/grpc/scheduler_service
RUN go mod download

WORKDIR /build
COPY database/ database/
COPY gen/go/ gen/go/
COPY calendar/ calendar/
COPY apps/grpc/scheduler_service/ apps/grpc/scheduler_service/

WORKDIR /build/apps/grpc/scheduler_service
RUN CGO_ENABLED=0 GOOS=linux go build -o /scheduler_service .

# --- Runtime stage ---
FROM alpine:3.21

RUN apk add --no-cache ca-certificates

RUN GRPC_HEALTH_PROBE_VERSION=v0.4.35 && \
    wget -qO/bin/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64 && \
    chmod +x /bin/grpc_health_probe

COPY --from=builder /scheduler_service /scheduler_service

EXPOSE 50051

ENTRYPOINT ["/scheduler_service"]
