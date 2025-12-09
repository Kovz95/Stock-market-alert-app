## PostgreSQL Migration – Phase 1

This document walks through the initial database setup on the VPS and applies
the schema that mirrors the current SQLite layout. No application code has been
pointed at Postgres yet – that happens in later phases.

---

### 1. Install PostgreSQL (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y postgresql postgresql-contrib
sudo systemctl enable --now postgresql
```

Check that it’s running:

```bash
sudo systemctl status postgresql
```

### 2. Create Role and Database

Replace `stockapp` and the password with whatever you prefer.

```bash
sudo -u postgres psql <<'SQL'
CREATE ROLE stockapp WITH LOGIN PASSWORD 'REPLACE_WITH_STRONG_PASSWORD';
CREATE DATABASE stockapp OWNER stockapp;
GRANT ALL PRIVILEGES ON DATABASE stockapp TO stockapp;
SQL
```

Allow local connections (optional if you deploy app on the same box):

```bash
sudo -u postgres psql -d postgres -c "ALTER ROLE stockapp SET timezone TO 'UTC';"
```

If you need password auth over TCP (e.g. app connects via 127.0.0.1), edit
`/etc/postgresql/*/main/pg_hba.conf` and add:

```
host    stockapp    stockapp    127.0.0.1/32    md5
```

Reload Postgres if you change configs:

```bash
sudo systemctl reload postgresql
```

### 3. Apply the Schema

Upload `db/postgres_schema.sql` to the server (e.g. scp) and run:

```bash
psql "postgresql://stockapp:REPLACE_WITH_STRONG_PASSWORD@localhost:5432/stockapp" \
    -f db/postgres_schema.sql
```

You should see “CREATE TABLE” / “CREATE INDEX” statements succeed.

### 4. Copy SQLite Files to the VPS (for later import)

You will need all SQLite databases currently used by the app:

```
alerts.db
alert_audit.db
futures_price_data.db
price_data.db
```

Copy them to the server (e.g. `/opt/stock-alert-app/data/`):

```bash
scp alerts.db user@vps:/opt/stock-alert-app/data/
scp alert_audit.db user@vps:/opt/stock-alert-app/data/
scp futures_price_data.db user@vps:/opt/stock-alert-app/data/
scp price_data.db user@vps:/opt/stock-alert-app/data/
```

### 5. Environment Variable Stub

Once we switch the app to Postgres, we’ll read connection details from
`DATABASE_URL`. Add the following placeholder to `.env` on the server:

```
DATABASE_URL=postgresql://stockapp:REPLACE_WITH_STRONG_PASSWORD@localhost:5432/stockapp
```

### 6. Next Steps

The next phase will provide:

1. A migration script that transfers data from the SQLite files into Postgres.
2. Refactors so the collectors, schedulers, and Streamlit pages use the
   Postgres connection.
3. Testing/deployment instructions that restart the services against the new DB.

For now, ensure Postgres is up, the schema is applied, and the SQLite files are
available on the server. Leave the app pointed at SQLite until the code
refactors are ready.
