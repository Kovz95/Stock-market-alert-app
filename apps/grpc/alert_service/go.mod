module stockalert/apps/grpc/alert_service

go 1.24.0

require (
	github.com/jackc/pgx/v5 v5.8.0
	google.golang.org/grpc v1.73.0
	google.golang.org/protobuf v1.36.6
	stockalert/alert v0.0.0
	stockalert/calendar v0.0.0
	stockalert/database v0.0.0
	stockalert/discord v0.0.0
	stockalert/gen/go v0.0.0
	stockalert/indicator v0.0.0
)

require (
	github.com/jackc/pgpassfile v1.0.0 // indirect
	github.com/jackc/pgservicefile v0.0.0-20240606120523-5a60cdf6a761 // indirect
	github.com/jackc/puddle/v2 v2.2.2 // indirect
	github.com/markcheno/go-talib v0.0.0-20250114000313-ec55a20c902f // indirect
	golang.org/x/net v0.38.0 // indirect
	golang.org/x/sync v0.17.0 // indirect
	golang.org/x/sys v0.31.0 // indirect
	golang.org/x/text v0.29.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20250324211829-b45e905df463 // indirect
	stockalert/expr v0.0.0 // indirect
)

replace (
	stockalert/alert => ../../../alert
	stockalert/calendar => ../../../calendar
	stockalert/database => ../../../database
	stockalert/discord => ../../../discord
	stockalert/expr => ../../../expr
	stockalert/gen/go => ../../../gen/go
	stockalert/indicator => ../../../indicator
)
