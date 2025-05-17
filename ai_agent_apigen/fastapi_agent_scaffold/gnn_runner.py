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
        try:
            output = json.loads(result.stdout)
            assignments = output.get("assignments", [])
            df = pd.DataFrame(assignments)
            df.columns = df.columns.astype(str)
            if not isinstance(assignments, list):
                raise ValueError("GNN output 'assignments' is not a list")
            return pd.DataFrame(assignments)
        except Exception as e:
            print("Failed to parse GNN output:", e)
        return None
        # return pd.DataFrame(json.loads(result.stdout)["assignments"])
    except Exception as e:
        print("❌ GNN execution failed:", e)
        return None

def run_gnn_pipeline():
    try:
        print("Starting GNN pipeline...")
        MONGO_URI = os.environ.get("MONGO_URI", "mongodb://3.105.47.11:27017")
        client = pymongo.MongoClient(MONGO_URI)
        db = client["sna_database"]

        df_weights = pd.DataFrame(list(db.sna_weights.find({}, {"_id": 0})))
        df_disrespect = pd.DataFrame(list(db.raw_disrespect.find({}, {"_id": 0}))).dropna()
        df_feedback = pd.DataFrame(list(db.raw_feedback.find({}, {"_id": 0}))).dropna()
        df_friendship = pd.DataFrame(list(db.raw_friendship.find({}, {"_id": 0}))).dropna()
        df_influence = pd.DataFrame(list(db.raw_influential.find({}, {"_id": 0}))).dropna()
        df_advice = pd.DataFrame(list(db.raw_advice.find({}, {"_id": 0}))).dropna()

        edges = {
            "disrespect": df_disrespect[['source','target']].to_numpy().tolist(),
            "feedback": df_feedback[['source','target']].to_numpy().tolist(),
            "friendship": df_friendship[['source','target']].to_numpy().tolist(),
            "influence": df_influence[['source','target']].to_numpy().tolist(),
            "advice": df_advice[['source','target']].to_numpy().tolist()
        }

        views = [
            {
                "edges":     edges[rel],
                "weight":    df_weights[rel][0].item(),
                "view_type": rel
            }
            for rel in ["disrespect", "feedback", "friendship", "influence", "advice"]
        ]

        payload = {"views": views}
        cluster_df = generate_clusters(json.dumps(payload))

        if cluster_df is None or cluster_df.empty:
            print("❌ No clusters returned from GNN")
            return None

        # Force column names to string and validate presence
        cluster_df.columns = cluster_df.columns.astype(str)
        if "id" not in cluster_df.columns or "cluster" not in cluster_df.columns:
            print("❌ GNN output missing 'id' or 'cluster' columns")
            return None

        print("Cluster DF columns:", cluster_df.columns)
        print("Cluster DF preview:\n", cluster_df.head())

        # Save to Mongo
        db["gnn_results"].delete_many({})
        db["gnn_results"].insert_many(cluster_df.to_dict("records"))

        print("✅ GNN pipeline complete")
        return cluster_df

    except Exception as e:
        print("❌ GNN pipeline error:", str(e))
        return None
