module stockalert/apps/scheduler

go 1.24.0

require (
	github.com/hibiken/asynq v0.24.1
	github.com/jackc/pgx/v5 v5.8.0
	stockalert/alert v0.0.0
	stockalert/calendar v0.0.0
	stockalert/database v0.0.0
	stockalert/discord v0.0.0
	stockalert/indicator v0.0.0
)

require (
	github.com/cespare/xxhash/v2 v2.2.0 // indirect
	github.com/dgryski/go-rendezvous v0.0.0-20200823014737-9f7001d12a5f // indirect
	github.com/golang/protobuf v1.5.2 // indirect
	github.com/google/uuid v1.2.0 // indirect
	github.com/jackc/pgpassfile v1.0.0 // indirect
	github.com/jackc/pgservicefile v0.0.0-20240606120523-5a60cdf6a761 // indirect
	github.com/jackc/puddle/v2 v2.2.2 // indirect
	github.com/markcheno/go-talib v0.0.0-20250114000313-ec55a20c902f // indirect
	github.com/redis/go-redis/v9 v9.0.3 // indirect
	github.com/robfig/cron/v3 v3.0.1 // indirect
	github.com/spf13/cast v1.3.1 // indirect
	golang.org/x/sync v0.17.0 // indirect
	golang.org/x/sys v0.28.0 // indirect
	golang.org/x/text v0.29.0 // indirect
	golang.org/x/time v0.0.0-20190308202827-9d24e82272b4 // indirect
	google.golang.org/protobuf v1.26.0 // indirect
	stockalert/expr v0.0.0 // indirect
)

replace (
	stockalert/alert => ../../alert
	stockalert/calendar => ../../calendar
	stockalert/database => ../../database
	stockalert/discord => ../../discord
	stockalert/expr => ../../expr
	stockalert/indicator => ../../indicator
)
