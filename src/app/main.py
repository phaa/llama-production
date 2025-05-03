import time
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Image reading
from io import BytesIO
from PIL import Image

# Monitoring
from prometheus_client import start_http_server, Summary, Counter, Gauge
from prometheus_client import REGISTRY, generate_latest
from starlette.responses import Response

import model_utils as utils

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUEST_TIME = Summary('request_processing_seconds', 'Tempo de processamento das requisições')
PREDICTIONS_TOTAL = Counter('predictions_total', 'Número total de predições realizadas')

@app.get("/metrics")
def metrics():
    """
    - Acessed automatically by Prometheus
    """
    return Response(generate_latest(REGISTRY), media_type="text/plain")


@app.post("/check-vegetation")
async def check_vegetation(files: list[UploadFile]):
    with REQUEST_TIME.time():
        images = []
        for file in files:
            contents = await file.read()
            images.append(Image.open(BytesIO(contents)).convert("RGB"))

        result = await utils.check_vegetation(images)
        context = { "response": result }

        PREDICTIONS_TOTAL.inc()

        return JSONResponse(content=context)


@app.post("/classify-switches")
async def classify_switches(files: list[UploadFile]):
    with REQUEST_TIME.time():
        images = []
        for file in files:
            contents = await file.read()
            images.append(Image.open(BytesIO(contents)).convert("RGB"))

        result = await utils.recognize_pole_switch(images)
        context = { "response": result }

        PREDICTIONS_TOTAL.inc()

        return JSONResponse(content=context)
    

@app.post("/process-transformer")
async def process_transformer(file: UploadFile):
    with REQUEST_TIME.time():
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

        result = utils.recognize_pole_transformer(image)
        context = { "response": result }

        PREDICTIONS_TOTAL.inc()

        return JSONResponse(content=context)




@app.get("/")
def read_root():
    return {"Teste": "Teste"}
