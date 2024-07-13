import os
from pathlib import Path
from typing import Annotated
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from tempfile import NamedTemporaryFile, TemporaryDirectory

from model import classify_image, classify_image2

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict/")
async def predict(file: UploadFile):
    try:
        with NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Process the image using the provided path
        yhat, hat = classify_image(tmp_path)
        return {"filename": file.filename, "probablity": yhat, "final verdict": hat}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/get_image")
async def get_image():
    image_path = Path("M00027.png")
    if not image_path.is_file():
        return {"error": "Image not found on the server"}
    return FileResponse(image_path)


@app.post("/predict2/")
async def predict2(file: UploadFile):
    try:
        yhat, hat = classify_image2(file.file, file.filename)
        return {"filename": file.filename, "probability": yhat, "final verdict": hat}
    finally:
        file.file.close()