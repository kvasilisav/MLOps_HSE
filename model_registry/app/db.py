import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiosqlite


def get_db_path() -> str:
    return os.environ.get("REGISTRY_DB", "registry.db")


def get_storage_path() -> str:
    return os.environ.get("REGISTRY_STORAGE", "./models_storage")


async def init_db() -> None:
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        await db.executescript(
            """
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                team TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                original_filename TEXT,
                status TEXT NOT NULL DEFAULT 'staging',
                metadata TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE,
                UNIQUE(model_id, version)
            );
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_id INTEGER NOT NULL,
                tag_key TEXT NOT NULL,
                tag_value TEXT NOT NULL,
                FOREIGN KEY (version_id) REFERENCES model_versions(id) ON DELETE CASCADE
            );
            """
        )
        await db.commit()


@asynccontextmanager
async def db_connection() -> AsyncIterator[aiosqlite.Connection]:
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        db.row_factory = aiosqlite.Row
        yield db
