# --- main.py ---
import random
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
from ga_runner import run_ga_allocation
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
    allow_origins=["*"],  # Allow frontend URL
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
        df_disrespect = pd.DataFrame(list(db.raw_disrespect.find({}, {"_id": 0}))).dropna()
        df_feedback   = pd.DataFrame(list(db.raw_feedback.find({}, {"_id": 0}))).dropna()
        df_friendship = pd.DataFrame(list(db.raw_friendship.find({}, {"_id": 0}))).dropna()
        df_influence  = pd.DataFrame(list(db.raw_influential.find({}, {"_id": 0}))).dropna()
        df_advice     = pd.DataFrame(list(db.raw_advice.find({}, {"_id": 0}))).dropna()

        edges = {
            "Disrespect":      df_disrespect[['source','target']].to_numpy().tolist(),
            "Feedback":        df_feedback[['source','target']].to_numpy().tolist(),
            "Friends":         df_friendship[['source','target']].to_numpy().tolist(),
            "Influence":       df_influence[['source','target']].to_numpy().tolist(),
            "Advice":          df_advice[['source','target']].to_numpy().tolist()
        }

        flat_edges = []
        for label, pairs in edges.items():
            for src, tgt in pairs:
                flat_edges.append({
                    "source": int(src),
                    "target": int(tgt),
                    "label":  label
                })
        return flat_edges
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

        db["gnn_results"].delete_many({})
        db["gnn_results"].insert_many(clusters.to_dict("records"))

        return {"status": "success", "message": "GNN executed successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/run_ga")
async def run_ga():
    try:
        clusters = run_ga_allocation(perf_field = "Perc_Academic")
        if clusters is None:
            return {"status": "error", "message": "GA processing failed."}

        db["result_node_cluster"].delete_many({})
        db["result_node_cluster"].insert_many(clusters.to_dict("records"))

        return {"status": "success", "message": "GA executed successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

    # get nodes data

@app.api_route("/run_gnn_all", methods=["GET", "POST"])
async def run_all():
    try:
        # Step 1: Run GNN
        gnn_clusters = run_gnn_pipeline()
        if gnn_clusters is None:
            return {"status": "error", "step": "gnn", "message": "GNN processing failed."}

        # Step 2: Run GA (after GNN)
        ga_clusters = run_ga_allocation(perf_field="Perc_Academic")
        print(ga_clusters)
        if ga_clusters is None:
            return {"status": "error", "step": "ga", "message": "GA processing failed."}

        db["result_node_cluster"].delete_many({})
        db["result_node_cluster"].insert_many(ga_clusters)
        return {"status": "success", "message": "Both GNN and GA executed successfully."}

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/result_node_cluster")
def get_result_edges_info():
    try:
        collection = db["result_node_cluster"]
        documents = list(collection.find({}, {"_id": 0}))
        sanitized_docs = [sanitize(doc) for doc in documents]
        return sanitized_docs
    except Exception as e:
        return {"error": str(e)}

@app.post("/run_naive_allocate")
def naive_allocate():
    try:
        df_raw = pd.DataFrame(list(db.sna_student_raw.find({}, {"Participant-ID": 1, "First-Name": 1, "Last-Name": 1})))
        df_weights = pd.DataFrame(list(db.sna_weights.find({}, {"_id": 0})))
        id_name_map = {
            row["Participant-ID"]: f"{row.get('First-Name', '').strip()} {row.get('Last-Name', '').strip()}".strip()
            for _, row in df_raw.iterrows()
        }
        student_ids = list(id_name_map.keys())

        total_students = len(student_ids)
        num_classes = max(1, math.ceil(total_students / df_weights.classSize.iloc[0]))
        random.shuffle(student_ids)
        assignments = []
        for idx, sid in enumerate(student_ids):
            cluster_id = idx % num_classes
            assignments.append({
                "id": sid,
                "label": id_name_map.get(sid, ""),
                "cluster": cluster_id
            })
        db["result_node_cluster"].delete_many({})
        db["result_node_cluster"].insert_many(assignments)
        return {"status": "success", "message": "Naively allocated students"}
    except Exception as e:
        return {"error": str(e)}



@app.get("/routes")
def get_routes():
    return [route.path for route in app.routes]
@app.post("/upload_csv/{collection_name}")
async def upload_csv(collection_name: str, file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported.")

        contents = await file.read()
        print(f"📄 Received file: {file.filename} (Size: {len(contents)} bytes)")

        # Load the content as a string (safe decoding)
        decoded = contents.decode("utf-8", errors="replace")

        # Now read with Pandas
        from io import StringIO
        df = pd.read_csv(StringIO(decoded))
        print("🧮 DataFrame shape:", df.shape)
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
        db["sna_weights"].delete_many({})
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

# ✅ upload csv path by hassan
@app.post("/upload_raw_csv")
async def upload_raw_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))

        # 🔄 Replace these with your actual columns
        participant_df = df[["participant_id", "age", "gender", "school_id", "grade", "cluster"]]
        school_df = df[["school_id", "school_name", "location"]].drop_duplicates()
        advice_df = df[["participant_id", "advice_to_id", "advice_strength"]]
        feedback_df = df[["participant_id", "feedback_to_id", "feedback_strength"]]
        disrespect_df = df[["participant_id", "disrespect_to_id", "disrespect_strength"]]
        influence_df = df[["participant_id", "influence_to_id", "influence_strength"]]
        friendship_df = df[["participant_id", "friendship_to_id", "friendship_strength"]]

        tables = {
            "participant": participant_df,
            "school": school_df,
            "advice": advice_df,
            "feedback": feedback_df,
            "disrespect": disrespect_df,
            "influence": influence_df,
            "friendship": friendship_df
        }

        inserted_counts = {}
        for name, table_df in tables.items():
            table_df = table_df.where(pd.notnull(table_df), None)
            records = table_df.to_dict(orient="records")
            db[name].delete_many({})
            if records:
                db[name].insert_many(records)
                inserted_counts[name] = len(records)
            else:
                inserted_counts[name] = 0

        return {"status": "success", "inserted": inserted_counts}

    except Exception as e:
        print("❌ CSV Split Upload Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# new route for new agent
@app.post("/ask_agent")

async def ask_agent(user_prompt: str = Form(...)):
    print("User prompt received:", user_prompt)
    if not user_prompt:
        return {"error": "Missing prompt."}

    # Step 1: Get query plan from LLM
    query_plan = generate_query_plan(user_prompt)

    if "error" in query_plan:
        return {
            "error": "Failed to generate query plan from LLM.",
            "detail": query_plan.get("error"),
            "raw": query_plan.get("raw_response")
        }

    # Step 2: Execute MongoDB query
    collection = query_plan.get("collection")
    pipeline = query_plan.get("pipeline")
    filter_ = query_plan.get("filter")
    projection = query_plan.get("projection", [])
    sort = query_plan.get("sort")
    limit = query_plan.get("limit", 100)

    if not collection:
        return {"error": "Missing collection name in query plan."}

    try:
        if pipeline:
            # Aggregation query
            results = list(db[collection].aggregate(pipeline))
        else:
            # Simple find query
            proj_dict = {field: 1 for field in projection} if projection else {}
            cursor = db[collection].find(filter_ or {}, proj_dict)

            if sort:
                cursor = cursor.sort(sort)
            if limit:
                cursor = cursor.limit(limit)

            results = list(cursor)

        # Clean ObjectIds for JSON output
        results = [sanitize(doc) for doc in results]
        # for doc in results:
        #     doc["_id"] = str(doc.get("_id", ""))

        return {
            "explanation": query_plan.get("explanation", ""),
            "results": results
        }

    except Exception as e:
        return {
            "error": "Failed to execute MongoDB query.",
            "detail": str(e)
        }
# integrate code with the AI agent, the ai agent use this route
@app.post("/generate", response_class=HTMLResponse)
def generate(request: Request, user_prompt: str = Form(...)):
    # Get query plan from LLM
    query_plan = generate_query_plan(user_prompt)
    print("== QUERY PLAN ==")
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
    if not collection_name:
        return {"error": "Query plan did not include a valid collection name."}

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
from pydantic import BaseModel

class ClusterWeights(BaseModel):
    friendship: float
    influence: float
    feedback: float
    advice: float
    disrespect: float
    affiliation: float

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
