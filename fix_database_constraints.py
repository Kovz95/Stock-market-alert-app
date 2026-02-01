"""
Fix PostgreSQL database constraints for price tables
Adds PRIMARY KEY constraints if they're missing
"""

import logging
from db_config import db_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_constraint_exists(conn, table_name, constraint_name):
    """Check if a constraint exists on a table"""
    query = """
        SELECT COUNT(*)
        FROM information_schema.table_constraints
        WHERE table_name = %s AND constraint_name = %s
    """
    cursor = conn.execute(query, (table_name, constraint_name))
    result = cursor.fetchone()
    return result[0] > 0

def fix_daily_prices_table():
    """Fix the daily_prices table constraints"""
    logger.info("Checking daily_prices table...")

    with db_config.connection(role="price_data") as conn:
        # Check if primary key exists
        has_pk = check_constraint_exists(conn, 'daily_prices', 'daily_prices_pkey')

        if has_pk:
            logger.info("✓ daily_prices already has PRIMARY KEY constraint")
        else:
            logger.warning("✗ daily_prices is missing PRIMARY KEY constraint")
            logger.info("Adding PRIMARY KEY constraint...")

            try:
                # Add primary key constraint
                conn.execute("""
                    ALTER TABLE daily_prices
                    ADD CONSTRAINT daily_prices_pkey PRIMARY KEY (ticker, date)
                """)
                conn.commit()
                logger.info("✓ Successfully added PRIMARY KEY constraint to daily_prices")
            except Exception as e:
                logger.error(f"Failed to add PRIMARY KEY: {e}")
                # If there are duplicate rows, we need to clean them first
                if "duplicate" in str(e).lower() or "already exists" in str(e).lower():
                    logger.info("Removing duplicate rows...")
                    conn.rollback()

                    # Remove duplicates, keeping the most recent update
                    conn.execute("""
                        DELETE FROM daily_prices a
                        USING daily_prices b
                        WHERE a.ticker = b.ticker
                        AND a.date = b.date
                        AND a.updated_at < b.updated_at
                    """)
                    conn.commit()

                    # Try adding constraint again
                    conn.execute("""
                        ALTER TABLE daily_prices
                        ADD CONSTRAINT daily_prices_pkey PRIMARY KEY (ticker, date)
                    """)
                    conn.commit()
                    logger.info("✓ Successfully added PRIMARY KEY constraint after cleanup")
                else:
                    raise

def fix_weekly_prices_table():
    """Fix the weekly_prices table constraints"""
    logger.info("Checking weekly_prices table...")

    with db_config.connection(role="price_data") as conn:
        # Check if primary key exists
        has_pk = check_constraint_exists(conn, 'weekly_prices', 'weekly_prices_pkey')

        if has_pk:
            logger.info("✓ weekly_prices already has PRIMARY KEY constraint")
        else:
            logger.warning("✗ weekly_prices is missing PRIMARY KEY constraint")
            logger.info("Adding PRIMARY KEY constraint...")

            try:
                # Add primary key constraint
                conn.execute("""
                    ALTER TABLE weekly_prices
                    ADD CONSTRAINT weekly_prices_pkey PRIMARY KEY (ticker, week_ending)
                """)
                conn.commit()
                logger.info("✓ Successfully added PRIMARY KEY constraint to weekly_prices")
            except Exception as e:
                logger.error(f"Failed to add PRIMARY KEY: {e}")
                # If there are duplicate rows, we need to clean them first
                if "duplicate" in str(e).lower() or "already exists" in str(e).lower():
                    logger.info("Removing duplicate rows...")
                    conn.rollback()

                    # Remove duplicates, keeping the most recent update
                    conn.execute("""
                        DELETE FROM weekly_prices a
                        USING weekly_prices b
                        WHERE a.ticker = b.ticker
                        AND a.week_ending = b.week_ending
                        AND a.updated_at < b.updated_at
                    """)
                    conn.commit()

                    # Try adding constraint again
                    conn.execute("""
                        ALTER TABLE weekly_prices
                        ADD CONSTRAINT weekly_prices_pkey PRIMARY KEY (ticker, week_ending)
                    """)
                    conn.commit()
                    logger.info("✓ Successfully added PRIMARY KEY constraint after cleanup")
                else:
                    raise

def fix_hourly_prices_table():
    """Fix the hourly_prices table constraints"""
    logger.info("Checking hourly_prices table...")

    with db_config.connection(role="price_data") as conn:
        # Check if primary key exists
        has_pk = check_constraint_exists(conn, 'hourly_prices', 'hourly_prices_pkey')

        if has_pk:
            logger.info("✓ hourly_prices already has PRIMARY KEY constraint")
        else:
            logger.warning("✗ hourly_prices is missing PRIMARY KEY constraint")
            logger.info("Adding PRIMARY KEY constraint...")

            try:
                # Add primary key constraint
                conn.execute("""
                    ALTER TABLE hourly_prices
                    ADD CONSTRAINT hourly_prices_pkey PRIMARY KEY (ticker, datetime)
                """)
                conn.commit()
                logger.info("✓ Successfully added PRIMARY KEY constraint to hourly_prices")
            except Exception as e:
                logger.error(f"Failed to add PRIMARY KEY: {e}")
                # If there are duplicate rows, we need to clean them first
                if "duplicate" in str(e).lower() or "already exists" in str(e).lower():
                    logger.info("Removing duplicate rows...")
                    conn.rollback()

                    # Remove duplicates, keeping the most recent update
                    conn.execute("""
                        DELETE FROM hourly_prices a
                        USING hourly_prices b
                        WHERE a.ticker = b.ticker
                        AND a.datetime = b.datetime
                        AND a.updated_at < b.updated_at
                    """)
                    conn.commit()

                    # Try adding constraint again
                    conn.execute("""
                        ALTER TABLE hourly_prices
                        ADD CONSTRAINT hourly_prices_pkey PRIMARY KEY (ticker, datetime)
                    """)
                    conn.commit()
                    logger.info("✓ Successfully added PRIMARY KEY constraint after cleanup")
                else:
                    raise

def fix_ticker_metadata_table():
    """Fix the ticker_metadata table constraints"""
    logger.info("Checking ticker_metadata table...")

    with db_config.connection(role="price_data") as conn:
        # Check if primary key exists
        has_pk = check_constraint_exists(conn, 'ticker_metadata', 'ticker_metadata_pkey')

        if has_pk:
            logger.info("✓ ticker_metadata already has PRIMARY KEY constraint")
        else:
            logger.warning("✗ ticker_metadata is missing PRIMARY KEY constraint")
            logger.info("Adding PRIMARY KEY constraint...")

            try:
                # Add primary key constraint
                conn.execute("""
                    ALTER TABLE ticker_metadata
                    ADD CONSTRAINT ticker_metadata_pkey PRIMARY KEY (ticker)
                """)
                conn.commit()
                logger.info("✓ Successfully added PRIMARY KEY constraint to ticker_metadata")
            except Exception as e:
                logger.error(f"Failed to add PRIMARY KEY: {e}")
                # If there are duplicate rows, we need to clean them first
                if "duplicate" in str(e).lower() or "already exists" in str(e).lower():
                    logger.info("Removing duplicate rows...")
                    conn.rollback()

                    # Remove duplicates, keeping the most recent update
                    conn.execute("""
                        DELETE FROM ticker_metadata a
                        USING ticker_metadata b
                        WHERE a.ticker = b.ticker
                        AND a.last_update < b.last_update
                    """)
                    conn.commit()

                    # Try adding constraint again
                    conn.execute("""
                        ALTER TABLE ticker_metadata
                        ADD CONSTRAINT ticker_metadata_pkey PRIMARY KEY (ticker)
                    """)
                    conn.commit()
                    logger.info("✓ Successfully added PRIMARY KEY constraint after cleanup")
                else:
                    raise

def fix_app_documents_table():
    """Fix the app_documents table constraints"""
    logger.info("Checking app_documents table...")

    with db_config.connection() as conn:
        # Check if primary key exists
        has_pk = check_constraint_exists(conn, 'app_documents', 'app_documents_pkey')

        if has_pk:
            logger.info("✓ app_documents already has PRIMARY KEY constraint")
        else:
            logger.warning("✗ app_documents is missing PRIMARY KEY constraint")
            logger.info("Adding PRIMARY KEY constraint...")

            try:
                # Add primary key constraint
                conn.execute("""
                    ALTER TABLE app_documents
                    ADD CONSTRAINT app_documents_pkey PRIMARY KEY (document_key)
                """)
                conn.commit()
                logger.info("✓ Successfully added PRIMARY KEY constraint to app_documents")
            except Exception as e:
                logger.error(f"Failed to add PRIMARY KEY: {e}")
                # If there are duplicate rows, we need to clean them first
                if "duplicate" in str(e).lower() or "already exists" in str(e).lower():
                    logger.info("Removing duplicate rows...")
                    conn.rollback()

                    # Remove duplicates, keeping the most recent update
                    conn.execute("""
                        DELETE FROM app_documents a
                        USING app_documents b
                        WHERE a.document_key = b.document_key
                        AND a.updated_at < b.updated_at
                    """)
                    conn.commit()

                    # Try adding constraint again
                    conn.execute("""
                        ALTER TABLE app_documents
                        ADD CONSTRAINT app_documents_pkey PRIMARY KEY (document_key)
                    """)
                    conn.commit()
                    logger.info("✓ Successfully added PRIMARY KEY constraint after cleanup")
                else:
                    raise

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Fixing database constraints for all tables")
    logger.info("=" * 70)

    try:
        fix_daily_prices_table()
        fix_weekly_prices_table()
        fix_hourly_prices_table()
        fix_ticker_metadata_table()
        fix_app_documents_table()

        logger.info("=" * 70)
        logger.info("✓ All constraints have been fixed successfully!")
        logger.info("=" * 70)
    except Exception as e:
        logger.error(f"Failed to fix constraints: {e}", exc_info=True)
        exit(1)
