import subprocess
import json
import pandas as pd
import pymongo
import os
model_path = os.path.join(os.getcwd(), "gnnworker", "assets", "model.bson")
def generate_clusters(data: str):
    try:
        result = subprocess.run(
            [
                os.path.join(os.getcwd(), "gnnworker", "bin", "GNNProject"),
                "--stdin",
                "--model-path=" + model_path
            ],
            input=data,
            text=True,
            capture_output=True,
            env={**os.environ, "JULIA_NUM_THREADS": "auto"}
        )

        print("📤 stdout:", result.stdout)
        print("🐛 stderr:", result.stderr)
        return pd.DataFrame(json.loads(result.stdout)["assignments"])
    except Exception as e:
        print("❌ GNN execution failed:", e)
        return None

def run_gnn_pipeline():
    MONGO_URI = os.environ.get("MONGO_URI", "mongodb://3.105.47.11:27017")
    client = pymongo.MongoClient(MONGO_URI)
    db = client["sna_database"]
    VIEW_TYPE_MAP = {
        "Friends":      "friendship",
        "Influential":  "influence",
        "Feedback":     "feedback",
        "Advice":       "advice",
        "Disrespect":   "disrespect",
    }

    df_raw = pd.DataFrame(list(db.sna_student_raw.find({}, {"_id": 0})))
    colnames = df_raw.columns.tolist()
    df_weights = pd.DataFrame(list(db.sna_weights.find({}, {"_id": 0})))

    views = []
    for raw_view, renamed_view in VIEW_TYPE_MAP.items():
        src_col = f"Source {raw_view}"
        tgt_col = f"Target {raw_view}"

        if src_col in colnames and tgt_col in colnames:
            df_edges = df_raw[[src_col, tgt_col]].dropna()

            edges = [
                [int(src), int(tgt)]
                for src, tgt in zip(df_edges[src_col], df_edges[tgt_col])
                if pd.notna(src) and pd.notna(tgt)
            ]

            if edges:
                views.append({
                    "edges": edges,
                    "weight": df_weights[renamed_view][0].item(),
                    "view_type": renamed_view
                })

    cluster_df = generate_clusters(json.dumps({"views": views}))
    return cluster_df
