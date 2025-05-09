import os
import pandas as pd
import numpy as np
from collections import Counter
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from pymongo import MongoClient

# ——— 1) Connect & build name map ———
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://3.105.47.11:27017")
client    = MongoClient(MONGO_URI)
db        = client["sna_database"]

id_name_map = {}
for doc in db.sna_student_raw.find(
        {}, {"_id": 0, "Participant-ID": 1, "First-Name": 1, "Last-Name": 1}
    ):
    pid = doc.get("Participant-ID")
    if pid is None:
        continue
    pid   = int(pid)
    name  = f"{doc.get('First-Name','').strip()} {doc.get('Last-Name','').strip()}".strip()
    id_name_map[pid] = name

# ——— 2) Load clusters CSV & detect its label column ———
df_clusters = pd.read_csv("clustered_students.csv")
cols = df_clusters.columns.tolist()

# find student-ID column (case-insensitive)
sid_cols = [c for c in cols if c.lower() == "student_id"]
if not sid_cols:
    raise ValueError("No 'student_id' column found in clustered_students.csv")
sid_col = sid_cols[0]

# try to pick anything with “cluster” in its name
cluster_candidates = [
    c for c in cols
    if "cluster" in c.lower() and c.lower() != sid_col.lower()
]
if cluster_candidates:
    cluster_col = cluster_candidates[0]
else:
    # fallback: first column that isn’t student_id or an "index"
    fallback = [
        c for c in cols
        if c.lower() != sid_col.lower() and "index" not in c.lower()
    ]
    if not fallback:
        raise ValueError(
            "Couldn’t detect cluster column. I see only these columns:\n"
            f"  {cols}"
        )
    cluster_col = fallback[0]

# rename for consistency
df_clusters = df_clusters.rename(columns={cluster_col: "cluster_label"})
df_clusters[sid_col]        = pd.to_numeric(df_clusters[sid_col],        errors="coerce").astype(int)
df_clusters["cluster_label"] = pd.to_numeric(df_clusters["cluster_label"], errors="coerce").astype(int)

# number of distinct classes to force in GA
K = df_clusters["cluster_label"].nunique()

# ——— 3) Pull raw survey + performance ———
df_raw = pd.DataFrame(list(db.sna_student_raw.find({}, {"_id": 0})))
if "Participant-ID" in df_raw:
    df_raw["Participant-ID"] = pd.to_numeric(
        df_raw["Participant-ID"], errors="coerce"
    ).astype(int)

perf_field = "Perc_Academic July"
if perf_field not in df_raw.columns:
    raise KeyError(f"Missing '{perf_field}' in sna_student_raw")

df_raw[perf_field] = pd.to_numeric(df_raw[perf_field], errors="coerce")

df = (
    df_clusters[[sid_col]]
      .merge(
          df_raw[["Participant-ID", perf_field]]
            .rename(columns={
                "Participant-ID": sid_col,
                perf_field:       "Perc_Academic"
            }),
          on=sid_col,
          how="left"
      )
)
# avoid pandas FutureWarning
df["Perc_Academic"] = df["Perc_Academic"].fillna(df["Perc_Academic"].mean())

# ——— 4) GA setup ———
student_ids = df[sid_col].tolist()
perf_array  = df["Perc_Academic"].values.astype(float)
N           = len(student_ids)

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual",   list,        fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_group", np.random.randint, 0, K)
toolbox.register("individual",
                 tools.initRepeat,
                 creator.Individual,
                 toolbox.attr_group,
                 n=N)

def eval_assignment(individual):
    sizes = np.array(list(Counter(individual).values()), dtype=float)
    size_balance = float(np.std(sizes))

    groups = {}
    for idx, g in enumerate(individual):
        groups.setdefault(g, []).append(idx)
    class_means = [np.mean(perf_array[idxs]) for idxs in groups.values() if idxs]
    worst_mean = float(np.min(class_means))

    return size_balance, worst_mean

toolbox.register("evaluate", eval_assignment)
toolbox.register("mate",     tools.cxTwoPoint)
toolbox.register("mutate",   tools.mutUniformInt, low=0, up=K-1, indpb=0.05)
toolbox.register("select",   tools.selNSGA2)

# ——— 5) Run GA & persist ———
def main(seed=42):
    np.random.seed(seed)
    pop = [toolbox.individual() for _ in range(200)]
    pop = tools.selNSGA2(pop, k=len(pop))

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg_size_balance", lambda vals: np.mean([f[0] for f in vals]))
    stats.register("avg_worst_mean",    lambda vals: np.mean([f[1] for f in vals]))

    pop, logbook = algorithms.eaMuPlusLambda(
        population= pop,
        toolbox=    toolbox,
        mu=         200,
        lambda_=    400,
        cxpb=       0.6,
        mutpb=      0.3,
        ngen=       50,
        stats=      stats,
        verbose=    True
    )

    # local outputs
    pd.DataFrame(logbook).to_csv("ga_progress.csv", index=False)
    pareto = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
    best   = max(pareto, key=lambda ind: ind.fitness.values[1])

    # → Print class sizes
    counts = Counter(best)
    print("Class sizes:")
    for cls in sorted(counts):
        print(f"  Class {cls:2d}: {counts[cls]} students")

    pd.DataFrame({
        sid_col:      student_ids,
        "cluster_label": best
    }).to_csv("ga_allocation_solution.csv", index=False)

    df_log = pd.DataFrame(logbook)
    plt.figure(figsize=(6,4))
    plt.plot(df_log["gen"], df_log["avg_worst_mean"], label="Avg Worst-Class Mean")
    plt.xlabel("Generation")
    plt.ylabel("Worst-Class Mean")
    plt.title("Improvement of Worst-Class Mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig("academic_improvement.png")

    # write back to MongoDB
    target = db["result_node_cluster"]  # or "results_node_clusters"
    target.delete_many({})

    docs = [{
        "id":             int(sid),
        "label":          id_name_map.get(int(sid), ""),
        "cluster_number": int(grp)
    } for sid, grp in zip(student_ids, best)]
    target.insert_many(docs)
    print(f"Inserted {len(docs)} docs into '{target.name}'.")

if __name__ == "__main__":
    main()
