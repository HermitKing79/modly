"""Local AMD image constraints only; no generation or persistent image storage."""
from fastapi import APIRouter, File, UploadFile
from starlette.concurrency import run_in_threadpool

from services.amd_image_test import MAX_BYTES, test_image

router = APIRouter(tags=['amd'])


@router.post('/image-test')
async def image_test(image: UploadFile = File(...)):
    try:
        data = await image.read(MAX_BYTES+1)
        return await run_in_threadpool(test_image, data)
    finally:
        await image.close()
