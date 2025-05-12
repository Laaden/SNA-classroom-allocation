# from fastapi import FastAPI, HTTPException, Request, Form
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from pydantic import BaseModel
# from typing import List
# import pymongo
# import os
# import math
#
# # --- NEW: Import LLM Agent ---
# from llm_assistant import interpret_user_intent
#
# # --- CONFIGURATION ---
# MONGO_URI = os.getenv("MONGO_URI")
# DB_NAME = os.getenv("DB_NAME")
#
# # --- INIT ---
# app = FastAPI(title="FastAPI Generator Agent")
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")
#
# # --- DATABASE CONNECTION ---
# client = pymongo.MongoClient(MONGO_URI)
# db = client[DB_NAME]
#
# # function to clean the nan values
# def sanitize(doc):
#     for key, value in doc.items():
#         if isinstance(value, float) and math.isnan(value):
#             doc[key] = None
#     return doc
#
# # --- ROUTES ---
# @app.get("/", response_class=HTMLResponse)
# def index(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})
#
# @app.get("/ClassB/test_query")
# def test_query():
#     data = list(db['sna_student_raw'].find({}, {"_id": 0, "Attendance": 1, "bullying": 1}))
#     for doc in data:
#         doc.pop('_id', None)
#         sanitize(doc)
#     print("Total returned:", len(data))
#     return data
#
# @app.post("/generate", response_class=HTMLResponse)
# def generate_endpoint_from_prompt(request: Request, user_prompt: str = Form(...)):
#     # Step 1: Use AI agent to interpret user intent
#     result = interpret_user_intent(user_prompt)  # e.g. {'table': 'products', 'fields': ['name', 'price'], 'endpoint': 'basic_info'}
#
#     # 🔍 Debug print
#     print("Prompt:", user_prompt)
#     print("LLM output:", result)
#
#     # Step 2: Extract details
#     collection_name = result['table']
#     fields_list = result['fields']
#     endpoint_name = result['endpoint']
#
#     route_path = f"/{collection_name}/{endpoint_name}"
#
#     # Step 3: Inject endpoint dynamically
#     @app.get(route_path)
#     def dynamic_query():
#         filter_query = result.get("filter", {})
#         limit = result.get("limit") or 500
#         print("Returned fields:", fields_list)
#
#         if fields_list == ["*"]:
#             projection = None  # means return all fields
#             print("path 1")
#         else:
#             projection = {field: 1 for field in fields_list}
#             print("path 2")
#         if projection is None:
#             cursor = db[collection_name].find(filter_query)
#             print("path 3")
#         else:
#             cursor = db[collection_name].find(filter_query, projection)
#             print("path 4")
#         # cursor = db[collection_name].find(filter_query, projection or {})
#         if limit > 0:
#             cursor = cursor.limit(limit)
#             print("path 5")
#
#         data = list(cursor)
#         for doc in data:
#             doc.pop("_id", None)
#             sanitize(doc)
#
#         print(f"Returned {len(data)} records for endpoint: {route_path}")
#         return data
#
#     return templates.TemplateResponse("result.html", {
#         "request": request,
#         "url": route_path,
#         "fields": fields_list,
#         "collection": collection_name
#     })
# --- main.py ---
from fastapi import FastAPI, HTTPException, Request, Form, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pymongo
import os
import math
from llm_assistant import generate_query_plan

# Load environment variables
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

# --- INIT ---
app = FastAPI(title="LLM-Driven MongoDB API")
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

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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
