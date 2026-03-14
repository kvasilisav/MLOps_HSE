import atexit
import os
import shutil
import tempfile
from pathlib import Path

_tmpdir = tempfile.mkdtemp(prefix="registry_test_")
os.environ["REGISTRY_DB"] = os.path.join(_tmpdir, "test.db")
os.environ["REGISTRY_STORAGE"] = os.path.join(_tmpdir, "storage")


def _cleanup():
    shutil.rmtree(_tmpdir, ignore_errors=True)


atexit.register(_cleanup)


def _reset_db_and_storage():
    import asyncio

    from app.db import get_db_path, get_storage_path, init_db

    async def _init_and_clear():
        await init_db()
        import aiosqlite

        async with aiosqlite.connect(get_db_path()) as db:
            await db.execute("DELETE FROM tags")
            await db.execute("DELETE FROM model_versions")
            await db.execute("DELETE FROM models")
            await db.commit()

    asyncio.run(_init_and_clear())
    storage = Path(get_storage_path())
    if storage.exists():
        for item in storage.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink(missing_ok=True)
