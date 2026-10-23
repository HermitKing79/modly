"""Isolated local CPU image preview API; no application or generation routes."""
import os

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from routers.amd_image_test import router as image_router
from services.amd_runtime import absolute_file


def _configuration():
    nonce = os.environ.get('MODLY_AMD_PREVIEW_NONCE', '')
    if not nonce.strip() or len(nonce) > 256 or any(char in nonce for char in ('\0', '\r', '\n')):
        raise HTTPException(503, 'Preview identity is not configured')
    try:
        executable = absolute_file(os.environ.get('MODLY_AMD_IMAGE_TEST_PYTHON', ''), 'Preview CPU Python')
        if executable.name.lower() != 'python.exe':
            raise ValueError('Explicit Windows Python executable required')
    except (OSError, ValueError):
        raise HTTPException(503, 'Existing local CPU image runtime is not configured') from None
    return nonce


async def _configured():
    _configuration()


app = FastAPI(title='AMD Image Test Preview', docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(CORSMiddleware, allow_origins=['null'], allow_methods=['GET', 'POST'],
                   allow_headers=['Content-Type'], allow_credentials=False)


@app.get('/health')
async def health():
    return dict(status='ok', preview_nonce=_configuration(), generation_enabled=False)


app.include_router(image_router, prefix='/amd', dependencies=[Depends(_configured)])
