import subprocess
import json
import pandas as pd

def generate_clusters(data: str):
    out = subprocess.run(
        [
            "build/GNNWorker/bin/GNNProject",
            "--stdin",
            "--model-path=build/GNNWorker/assets/model.bson"
        ],
        input = data,
        text=True,
        capture_output=True
    ).stdout
    return pd.DataFrame(json.loads(out)["assignments"])


with open("scripts/test_input.json") as json_file:
    json_data = json.load(json_file)

clusters = generate_clusters(json.dumps(json_data))
print(clusters)