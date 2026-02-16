#!/usr/bin/env python3
"""
Fix alert_audits sequence out of sync with data.

The PostgreSQL BIGSERIAL sequence can get out of sync when data is
migrated/imported with explicit IDs. This script resets the sequence
to the maximum existing ID + 1.

Usage:
    python scripts/maintenance/fix_alert_audits_sequence.py
"""

import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data_access.db_config import db_config


def fix_sequence():
    """Reset the alert_audits_id_seq to max(id) + 1."""
    if db_config.db_type != "postgresql":
        print("This script only applies to PostgreSQL databases.")
        return

    with db_config.connection(role="alerts") as conn:
        cursor = conn.cursor()
        try:
            # Get current max ID
            cursor.execute("SELECT COALESCE(MAX(id), 0) FROM alert_audits")
            max_id = cursor.fetchone()[0]
            print(f"Current max ID in alert_audits: {max_id}")

            # Get current sequence value
            cursor.execute("SELECT last_value FROM alert_audits_id_seq")
            seq_value = cursor.fetchone()[0]
            print(f"Current sequence value: {seq_value}")

            if seq_value <= max_id:
                # Reset sequence to max_id + 1
                new_value = max_id + 1
                cursor.execute(
                    f"SELECT setval('alert_audits_id_seq', {new_value}, false)"
                )
                conn.commit()
                print(f"Sequence reset to {new_value}")
                print("Fix applied successfully!")
            else:
                print("Sequence is already ahead of max ID - no fix needed.")

        finally:
            cursor.close()


if __name__ == "__main__":
    fix_sequence()
