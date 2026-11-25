"""OncoSense AI — FastAPI application entry point.

Run (from backend/):  uvicorn app.main:app --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core import config
from app.services.registry import ArtifactsUnavailable, registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("oncosense")

# Multipart overhead allowance on top of the file-size limit
MAX_REQUEST_BYTES = config.MAX_UPLOAD_BYTES + 64_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.load()  # load trusted artifacts once; never train inside a request
    yield


app = FastAPI(
    title="OncoSense AI API",
    version="1.0.0",
    description=f"{config.DISCLAIMER} Educational breast-cancer classification research API "
                "built on the Breast Cancer Wisconsin (Diagnostic) dataset.",
    lifespan=lifespan,
    debug=False,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def guard_and_headers(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_REQUEST_BYTES:
        return _error(413, "payload_too_large", f"Request body exceeds {MAX_REQUEST_BYTES} bytes")
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = response.headers.get("Cache-Control", "no-store")
    return response


def _error(status: int, code: str, message: str, details=None) -> JSONResponse:
    return JSONResponse(status_code=status,
                        content={"error": {"code": code, "message": message, "details": details}})


@app.exception_handler(ArtifactsUnavailable)
async def _unavailable(_: Request, exc: ArtifactsUnavailable):
    return _error(503, "model_unavailable", "Model artifacts are not loaded. Run the training "
                  "script (python scripts/train_models.py) and restart the API.", str(exc))


@app.exception_handler(RequestValidationError)
async def _validation(_: Request, exc: RequestValidationError):
    details = [{"loc": [str(p) for p in e.get("loc", [])], "msg": e.get("msg"), "type": e.get("type")}
               for e in exc.errors()]
    return _error(422, "validation_error", "Input failed validation; no prediction was made.", details)


@app.exception_handler(HTTPException)
async def _http(_: Request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        return _error(exc.status_code, "request_error", exc.detail.get("message", "Error"), exc.detail)
    return _error(exc.status_code, "request_error", str(exc.detail))


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception):
    log.exception("Unhandled error")
    return _error(500, "internal_error", "An internal error occurred.")  # no stack traces to clients


app.include_router(router)

config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/figures", StaticFiles(directory=config.FIGURES_DIR), name="figures")
