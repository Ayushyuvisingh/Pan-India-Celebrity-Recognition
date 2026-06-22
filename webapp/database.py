import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

DB_NAME = "history.db"


def init_db():

    conn = sqlite3.connect(
        DB_NAME
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS recognition_history (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

            image_name TEXT,

            prediction TEXT,

            confidence REAL,

            mode TEXT

        )
        """
    )

    conn.commit()

    conn.close()


def save_prediction(
    image_name,
    prediction,
    confidence,
    mode
):
    
    current_time = datetime.now(
        ZoneInfo("Asia/Kolkata")
    ).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    conn = sqlite3.connect(
        DB_NAME
    )

    cursor = conn.cursor()

    

    cursor.execute(
        """
        INSERT INTO recognition_history (

            timestamp,
            image_name,
            prediction,
            confidence,
            mode

        )

        VALUES (?, ?, ?, ?, ?)
        """,
        (
            current_time,
            image_name,
            prediction,
            confidence,
            mode
        )
    )

    conn.commit()

    conn.close()


def get_history(
    page=1,
    per_page=20
):

    offset = (
        page - 1
    ) * per_page

    conn = sqlite3.connect(DB_NAME)

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM recognition_history
        ORDER BY id DESC
        LIMIT ?
        OFFSET ?
        """,
        (
            per_page,
            offset
        )
    )

    rows = cursor.fetchall()

    conn.close()

    return rows

def get_total_records():

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    total = cursor.execute(
        """
        SELECT COUNT(*)
        FROM recognition_history
        """
    ).fetchone()[0]

    conn.close()

    return total

def get_stats():

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    total = cursor.execute(
        """
        SELECT COUNT(*)
        FROM recognition_history
        """
    ).fetchone()[0]

    fast = cursor.execute(
        """
        SELECT COUNT(*)
        FROM recognition_history
        WHERE mode='FAST'
        """
    ).fetchone()[0]

    verified = cursor.execute(
        """
        SELECT COUNT(*)
        FROM recognition_history
        WHERE mode='VERIFIED'
        """
    ).fetchone()[0]

    unknown = cursor.execute(
        """
        SELECT COUNT(*)
        FROM recognition_history
        WHERE mode='UNKNOWN'
        """
    ).fetchone()[0]

    conn.close()

    return {

        "total": total,

        "fast": fast,

        "verified": verified,

        "unknown": unknown
    }