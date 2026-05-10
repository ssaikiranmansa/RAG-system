import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor


logger = logging.getLogger(__name__)


def get_conn():
    try:
        return psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=os.environ.get("POSTGRES_PORT", 5432),
            dbname=os.environ.get("POSTGRES_DB", "ragdb"),
            user=os.environ.get("POSTGRES_USER", "rag"),
            password=os.environ.get("POSTGRES_PASSWORD", "rag"),
        )
    except Exception as e:
        logger.error("Failed to connect to Postgres: %s", e)
        raise


def init_db():
    logger.info("Initialising database schema...")
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chunks (
                        id SERIAL PRIMARY KEY,
                        source TEXT,
                        chunk_index INTEGER,
                        content TEXT,
                        embedding vector(3072),
                        page_number INTEGER,
                        section_header TEXT,
                        word_count INTEGER
                    );
                """)
                # add columns if table already exists but columns are missing
                cur.execute("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='chunks' AND column_name='page_number'
                        ) THEN
                            ALTER TABLE chunks ADD COLUMN page_number INTEGER;
                        END IF;

                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='chunks' AND column_name='section_header'
                        ) THEN
                            ALTER TABLE chunks ADD COLUMN section_header TEXT;
                        END IF;

                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='chunks' AND column_name='word_count'
                        ) THEN
                            ALTER TABLE chunks ADD COLUMN word_count INTEGER;
                        END IF;
                    END
                    $$;
                """)
            conn.commit()
        logger.info("Database schema ready.")
    except Exception as e:
        logger.error("Database initialisation failed: %s", e)
        raise