import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.db import db_connection, get_storage_path, init_db


def _model_id(team: str, name: str) -> str:
    return f"{team}_{name}"


def _safe_filename(name: str) -> str:
    base = os.path.basename(name) or "artifact.bin"
    base = re.sub(r"[^\w.\-]", "_", base)
    return base or "artifact.bin"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    storage = Path(get_storage_path())
    storage.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Model Registry", lifespan=lifespan)


class StatusUpdate(BaseModel):
    status: str


@app.post("/models")
async def register_model(
    name: str = Form(...),
    team: str = Form(...),
    file: UploadFile = File(...),
    metadata: str | None = Form(None),
    tags: str | None = Form(None),
    status: str | None = Form(None),
):
    model_pk = _model_id(team, name)
    effective_status = (status or "staging").strip() or "staging"

    meta_obj: dict[str, Any] | None = None
    if metadata:
        try:
            meta_obj = json.loads(metadata)
            if not isinstance(meta_obj, dict):
                raise ValueError("metadata must be a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"invalid metadata JSON: {e}") from e

    tags_obj: dict[str, Any] | None = None
    if tags:
        try:
            tags_obj = json.loads(tags)
            if not isinstance(tags_obj, dict):
                raise ValueError("tags must be a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"invalid tags JSON: {e}") from e

    content = await file.read()
    fname = _safe_filename(file.filename or "model.bin")

    storage_root = Path(get_storage_path())

    async with db_connection() as db:
        async with db.execute(
            "SELECT COALESCE(MAX(version), 0) AS v FROM model_versions WHERE model_id = ?",
            (model_pk,),
        ) as cur:
            row = await cur.fetchone()
        next_ver = int(row["v"]) + 1

        rel_dir = Path(model_pk) / str(next_ver)
        dest_dir = storage_root / rel_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / fname
        dest_path.write_bytes(content)
        file_path = str(dest_path.resolve())

        await db.execute(
            "INSERT OR IGNORE INTO models (id, name, team) VALUES (?, ?, ?)",
            (model_pk, name, team),
        )
        cur = await db.execute(
            """
            INSERT INTO model_versions (model_id, version, file_path, original_filename, status, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                model_pk,
                next_ver,
                file_path,
                fname,
                effective_status,
                json.dumps(meta_obj) if meta_obj is not None else None,
            ),
        )
        await db.commit()
        version_row_id = cur.lastrowid

        if tags_obj and version_row_id is not None:
            for k, v in tags_obj.items():
                await db.execute(
                    "INSERT INTO tags (version_id, tag_key, tag_value) VALUES (?, ?, ?)",
                    (version_row_id, str(k), json.dumps(v)),
                )
            await db.commit()

    rel_path = str(rel_dir / fname).replace("\\", "/")
    return {"model_id": model_pk, "version": next_ver, "path": rel_path}


@app.get("/models")
async def list_models(team: str | None = None):
    async with db_connection() as db:
        if team:
            async with db.execute(
                "SELECT id, name, team FROM models WHERE team = ? ORDER BY created_at DESC",
                (team,),
            ) as cur:
                rows = await cur.fetchall()
        else:
            async with db.execute(
                "SELECT id, name, team FROM models ORDER BY created_at DESC",
            ) as cur:
                rows = await cur.fetchall()
    return [dict(r) for r in rows]


@app.get("/models/{model_id}")
async def get_model(model_id: str):
    async with db_connection() as db:
        async with db.execute(
            "SELECT id, name, team FROM models WHERE id = ?",
            (model_id,),
        ) as cur:
            m = await cur.fetchone()
        if not m:
            raise HTTPException(status_code=404, detail="model not found")

        async with db.execute(
            """
            SELECT id, version, status, metadata, original_filename, created_at
            FROM model_versions
            WHERE model_id = ?
            ORDER BY version ASC
            """,
            (model_id,),
        ) as cur:
            vrows = await cur.fetchall()

        tags_by_version: dict[int, dict[str, Any]] = {}
        vids = [int(r["id"]) for r in vrows]
        if vids:
            placeholders = ",".join("?" * len(vids))
            async with db.execute(
                f"SELECT version_id, tag_key, tag_value FROM tags WHERE version_id IN ({placeholders})",
                vids,
            ) as cur:
                for tr in await cur.fetchall():
                    vid = int(tr["version_id"])
                    tags_by_version.setdefault(vid, {})
                    try:
                        tags_by_version[vid][tr["tag_key"]] = json.loads(tr["tag_value"])
                    except json.JSONDecodeError:
                        tags_by_version[vid][tr["tag_key"]] = tr["tag_value"]

    versions_out: list[dict[str, Any]] = []
    for vr in vrows:
        meta_parsed = None
        if vr["metadata"]:
            try:
                meta_parsed = json.loads(vr["metadata"])
            except json.JSONDecodeError:
                meta_parsed = vr["metadata"]
        vid = int(vr["id"])
        versions_out.append(
            {
                "version": vr["version"],
                "status": vr["status"],
                "metadata": meta_parsed,
                "tags": tags_by_version.get(vid, {}),
                "filename": vr["original_filename"],
                "created_at": vr["created_at"],
            }
        )

    return {
        "id": m["id"],
        "name": m["name"],
        "team": m["team"],
        "versions": versions_out,
    }


@app.get("/models/{model_id}/versions/{version}")
async def download_version(model_id: str, version: int):
    async with db_connection() as db:
        async with db.execute(
            "SELECT file_path FROM model_versions WHERE model_id = ? AND version = ?",
            (model_id, version),
        ) as cur:
            row = await cur.fetchone()
    if not row or not os.path.isfile(row["file_path"]):
        raise HTTPException(status_code=404, detail="version not found")
    return FileResponse(
        path=row["file_path"],
        filename=os.path.basename(row["file_path"]),
        media_type="application/octet-stream",
    )


@app.patch("/models/{model_id}/versions/{version}")
async def patch_version_status(model_id: str, version: int, body: StatusUpdate):
    async with db_connection() as db:
        cur = await db.execute(
            """
            UPDATE model_versions SET status = ?
            WHERE model_id = ? AND version = ?
            """,
            (body.status, model_id, version),
        )
        await db.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="version not found")
    return {"ok": True}
