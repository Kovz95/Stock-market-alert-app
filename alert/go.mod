module stockalert/alert

go 1.24.0

require (
	github.com/jackc/pgx/v5 v5.8.0
	stockalert/database v0.0.0
	stockalert/expr v0.0.0
	stockalert/indicator v0.0.0
)

require (
	github.com/jackc/pgpassfile v1.0.0 // indirect
	github.com/jackc/pgservicefile v0.0.0-20240606120523-5a60cdf6a761 // indirect
	github.com/markcheno/go-talib v0.0.0-20250114000313-ec55a20c902f // indirect
	golang.org/x/text v0.29.0 // indirect
)

replace (
	stockalert/database => ../database
	stockalert/expr => ../expr
	stockalert/indicator => ../indicator
)
