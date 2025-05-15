# --- main.py ---
from fastapi import FastAPI, HTTPException, Request, Form, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import pymongo
import os
import math
from fastapi import UploadFile, File, Request
from fastapi.responses import RedirectResponse
import pandas as pd
from llm_assistant import generate_query_plan
import json

# Load environment variables
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

# --- HTML Form Page ---
# @app.get("/upload", response_class=HTMLResponse)
# def upload_page(request: Request):
#     return templates.TemplateResponse("upload.html", {"request": request})

# --- CSV Upload Endpoint ---
from fastapi import UploadFile, File, HTTPException
import pandas as pd


# --- INIT ---
app = FastAPI(title="LLM-Driven MongoDB API")

# Add CORS middleware to allow specific origins
origins = [
    "https://dannythesober.github.io",  # Add your frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- DATABASE CONNECTION ---
client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]

# --- Utility ---
from bson import ObjectId

def sanitize(doc):
    for key, value in doc.items():
        if isinstance(value, float) and math.isnan(value):
            doc[key] = None
        elif isinstance(value, ObjectId):
            doc[key] = str(value)
    return doc

# --- ROUTES ---
# get edges data
@app.get("/api/result_edges_info")
def get_result_edges_info():
    try:
        collection = db["result_edges_info"]
        documents = list(collection.find({}, {"_id": 0}))
        sanitized_docs = [sanitize(doc) for doc in documents]
        return sanitized_docs
    except Exception as e:
        return {"error": str(e)}

# trigger GNNs model
from gnn_runner import run_gnn_pipeline

@app.get("/gnn_trigger", response_class=HTMLResponse)
async def show_trigger_page(request: Request):
    return templates.TemplateResponse("trigger_gnn.html", {"request": request})

import os

print("Current working directory:", os.getcwd())

@app.post("/run_gnn")
async def run_gnn():
    try:
        clusters = run_gnn_pipeline()
        if clusters is None:
            return {"status": "error", "message": "GNN processing failed."}

        # (Optional) Save to MongoDB
        db["gnn_results"].delete_many({})
        db["gnn_results"].insert_many(clusters.to_dict("records"))

        return {"status": "success", "message": "GNN executed successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

    # get nodes data
@app.get("/api/result_node_cluster")
def get_result_edges_info():
    try:
        collection = db["result_node_cluster"]
        documents = list(collection.find({}, {"_id": 0}))
        sanitized_docs = [sanitize(doc) for doc in documents]
        return sanitized_docs
    except Exception as e:
        return {"error": str(e)}

@app.get("/routes")
def get_routes():
    return [route.path for route in app.routes]
@app.post("/upload_csv/{collection_name}")
async def upload_csv(collection_name: str, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        print(f"📄 Received file: {file.filename} (Size: {len(contents)} bytes)")

        df = pd.read_csv(pd.io.common.BytesIO(contents))
        print("🧮 DataFrame loaded:")
        print(df.head())

        # Replace NaN with None safely
        records = df.where(pd.notnull(df), None).to_dict(orient="records")
        print(f"📦 Total records parsed: {len(records)}")

        result = db[collection_name].insert_many(records)
        print(f"✅ Inserted {len(result.inserted_ids)} documents into '{collection_name}'")

        return {"status": "success", "inserted": len(result.inserted_ids)}
    except Exception as e:
        print("❌ Upload error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# update weights factor
@app.get("/weights", response_class=HTMLResponse)
async def get_weights_form(request: Request):
    return templates.TemplateResponse("weights.html", {"request": request})


@app.post("/update_weights")
async def update_weights(
        friendship: int = Form(...),
        influence: int = Form(...),
        feedback: int = Form(...),
        advice: int = Form(...),
        disrespect: int = Form(...),
        affiliation: int = Form(...)
):
    try:
        db["sna_weights"].delete_many({})  # Optional: clear old data
        db["sna_weights"].insert_one({
            "friendship": friendship,
            "influence": influence,
            "feedback": feedback,
            "advice": advice,
            "disrespect": disrespect,
            "affiliation": affiliation
        })
        return {"status": "success", "message": "Weights updated."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# get all edges
@app.get("/api/edges")
def get_all_edges():
    try:
        documents = list(collection.find())
        return JSONResponse(content=dumps(documents), media_type="application/json")
    except Exception as e:
        return {"error": str(e)}


@app.post("/upload_json/{collection_name}")
async def upload_json(collection_name: str, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        data = json.loads(contents.decode("utf-8"))

        # Ensure it's a list of records
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            raise ValueError("JSON file must contain a list or a single JSON object.")

        result = db[collection_name].insert_many(data)
        return {"status": "success", "inserted": len(result.inserted_ids)}
    except Exception as e:
        print("❌ Upload JSON error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# integrate code with the AI agent, the ai agent use this route
@app.post("/generate", response_class=HTMLResponse)
def generate(request: Request, user_prompt: str = Form(...)):
    # Get query plan from LLM
    query_plan = generate_query_plan(user_prompt)

    print("LLM Plan:", query_plan)

    collection_name = query_plan.get("collection")
    fields = query_plan.get("projection")
    if not fields:
    # fallback if projection is empty but we have $exists filter (implied fields)
        fields = list(query_plan.get("filter", {}).get("$and", [{}])[0].keys())

    mongo_filter = query_plan.get("filter", {})
    # Ensure sort is in correct format: list of [field, direction]
    raw_sort = query_plan.get("sort", [])
    sort = []

    for item in raw_sort:
        if isinstance(item, list) and len(item) == 2:
            sort.append(tuple(item))
        elif isinstance(item, str):  # fallback: assume descending sort
            sort.append((item, -1))
    limit = query_plan.get("limit", 100)
    endpoint = query_plan.get("endpoint")
    if not endpoint:
        # Fallback to a default name based on timestamp or a counter
        import time
        endpoint = f"endpoint_{int(time.time())}"

    # Construct projection
    if not fields or fields == ["*"]:
        projection = None
    else:
        projection = {field: 1 for field in fields}
        if "ID" in fields:
            projection["_id"] = 1
        else:
            projection["_id"] = 0

    # 🔍 Debug: Immediate execution and print of query
    print(f"\n--- Executing MongoDB Preview Query ---")
    print("Collection:", collection_name)
    print("Projection:", projection)
    print("Filter:", mongo_filter)
    print("Sort:", sort)
    print("Limit:", limit)

    test_cursor = db[collection_name].find(mongo_filter, projection or {})
    if sort:
        test_cursor = test_cursor.sort(sort)
    if limit:
        test_cursor = test_cursor.limit(limit)

    preview_results = [sanitize(doc) for doc in test_cursor]
    print(f"Returned {len(preview_results)} records (preview)")
    for doc in preview_results[:5]:
        print(doc)

    route_path = f"/{collection_name}/{endpoint}"

    @app.get(route_path)
    def dynamic():
        cursor = db[collection_name].find(mongo_filter, projection or {})
        if sort:
            cursor = cursor.sort(sort)
        if limit:
            cursor = cursor.limit(limit)
        results = []
        for doc in cursor:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
            results = [sanitize(doc) for doc in cursor]
            # results.append(doc)
        return results

    return templates.TemplateResponse("result.html", {
        "request": request,
        "url": route_path,
        "fields": fields,
        "collection": collection_name
    })


# ~~~~~~~~~~ Cluster Generation ~~~~~~~~~~~~~~~ #

class ClusterWeights:
    friendship: int
    influence: int
    feedback: int
    advice: int
    disrespect: int
    affiliation: int

@app.post("/update_weights/", status_code=status.HTTP_204_NO_CONTENT)
def update_weights(weights: ClusterWeights):
    col = db.sna_weights
    first_doc = col.find_one()
    query = {"_id": first_doc["_id"]}
    col.update_one(query, {"$set": weights.dict()})
    return None

@app.get("/get_weights/", response_model=ClusterWeights)
def get_weights() -> ClusterWeights:
    doc = db.sna_weights.find_one({}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="No weights config found")
    return doc
