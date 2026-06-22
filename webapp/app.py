from fastapi import FastAPI
from fastapi import Request
from fastapi import UploadFile
from fastapi import File

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import shutil
import os
import uuid

from recognizer import recognize_face
from fastapi.responses import JSONResponse



app = FastAPI()

from database import (
    init_db,
    save_prediction,
    get_history,
    get_stats,
    get_total_records
)

init_db()


os.makedirs("static", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)

# --------------------------------
# STATIC
# --------------------------------

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

templates = Jinja2Templates(
    directory="templates"
)

UPLOAD_DIR = "static/uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

# --------------------------------
# HOME
# --------------------------------

@app.get("/")
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": None,
            "image_path": None
        }
    )

# --------------------------------
# PREDICT
# --------------------------------

@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...)
):

    filename = f"{uuid.uuid4()}.jpg"
    original_filename = file.filename

    save_path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    with open(
        save_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    result = recognize_face(
        save_path
    )

    save_prediction(

        original_filename,

        result["prediction"],

        result["confidence"],

        result["mode"]

    )

    result["prediction"] = (
        result["prediction"]
        .replace("_", " ")
    )

    for match in result["top_matches"]:

        match["name"] = (
            match["name"]
            .replace("_", " ")
        )

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": result,
            "image_path": "/" + save_path.replace("\\", "/")
        }
    )

@app.post("/api/recognize")
async def api_recognize(
    file: UploadFile = File(...)
):

    filename = f"{uuid.uuid4()}.jpg"

    save_path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    with open(
        save_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    result = recognize_face(
        save_path
    )

    return JSONResponse(
        content=result
    )

@app.get("/history")
def history(
    request: Request,
    page: int = 1
):

    per_page = 20

    rows = get_history(
        page,
        per_page
    )

    total_records = (
        get_total_records()
    )

    total_pages = (
        total_records
        + per_page
        - 1
    ) // per_page

    return templates.TemplateResponse(

        request=request,

        name="history.html",

        context={

            "rows": rows,

            "page": page,

            "total_pages": total_pages

        }
    )

@app.get("/dashboard")
def dashboard(
    request: Request
):

    stats = get_stats()

    rows = get_history()[:5]

    return templates.TemplateResponse(

        request=request,

        name="dashboard.html",

        context={

            "stats": stats,

            "rows": rows

        }
    )