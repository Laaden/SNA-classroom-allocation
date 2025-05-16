import os
import pandas as pd
import numpy as np
from collections import Counter
from deap import base, creator, tools, algorithms
from pymongo import MongoClient

def run_ga_allocation(
    mongo_uri: str = None,
    db_name: str = "sna_database",
    clusters_json: list[dict] = None,
    perf_field: str = "Perc_Academic July",
    edge_field: str = "Target School Activities",
    pop_size: int = 200,
    ngen: int = 50,
    cxpb: float = 0.6,
    mutpb: float = 0.3,
    seed: int = 42
):
    """
    1) Connect to MongoDB and build an ID→Name map
    2) Load cluster assignments from JSON list
    3) Pull raw survey + performance data, merge and fill missing
    4) Set up & run a multi-objective GA
    5) Write cluster and edge docs back to MongoDB with id fields
    6) Return cluster_docs and edge_docs as JSON
    """
    # Connect & name map
    if mongo_uri is None:
        mongo_uri = os.environ.get("MONGO_URI", "mongodb://3.105.47.11:27017")
    client = MongoClient(mongo_uri)
    db = client[db_name]

    # Build Participant-ID → Name map
    id_name_map = {}
    for doc in db.sna_student_raw.find({}, {"Participant-ID": 1, "First-Name": 1, "Last-Name": 1}):
        pid = doc.get("Participant-ID")
        if pid is None:
            continue
        name = f"{doc.get('First-Name','').strip()} {doc.get('Last-Name','').strip()}".strip()
        id_name_map[int(pid)] = name

    # Load cluster assignments (JSON list of {"id": student_id, "cluster": int})
    if clusters_json is None:
        clusters_json = list(db.result_node_cluster.find({}))
    df_clusters = pd.DataFrame(clusters_json)
    if "id" not in df_clusters.columns or "cluster" not in df_clusters.columns:
        raise ValueError("clusters_json must contain 'id' and 'cluster'")
    df_clusters = df_clusters.rename(columns={"id": "student_id"})
    df_clusters["student_id"] = pd.to_numeric(df_clusters["student_id"], errors="coerce").astype(int)
    df_clusters["cluster"] = pd.to_numeric(df_clusters["cluster"], errors="coerce").astype(int)
    K = df_clusters["cluster"].nunique()

    # Pull raw survey + performance
    df_raw = pd.DataFrame(list(db.sna_student_raw.find({}, {"Participant-ID": 1, perf_field: 1})))
    df_raw = df_raw.rename(columns={"Participant-ID": "student_id"})
    df_raw["student_id"] = pd.to_numeric(df_raw["student_id"], errors="coerce").astype(int)
    if perf_field not in df_raw.columns:
        raise KeyError(f"Missing '{perf_field}' in sna_student_raw")
    df_raw[perf_field] = pd.to_numeric(df_raw[perf_field], errors="coerce")

    df = (
        df_clusters[["student_id"]]
          .merge(
              df_raw[["student_id", perf_field]].rename(columns={perf_field: "Perc_Academic"}),
              on="student_id", how="left"
          )
    )
    df["Perc_Academic"] = df["Perc_Academic"].fillna(df["Perc_Academic"].mean())

    student_ids = df["student_id"].tolist()
    perf_array = df["Perc_Academic"].values.astype(float)
    N = len(student_ids)

    # GA setup
    try:
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    except RuntimeError:
        pass

    toolbox = base.Toolbox()
    toolbox.register("attr_group", np.random.randint, 0, K)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_group, n=N)

    def eval_assignment(ind):
        sizes = np.array(list(Counter(ind).values()), dtype=float)
        size_balance = float(np.std(sizes))
        groups = {}
        for idx, g in enumerate(ind): groups.setdefault(g, []).append(idx)
        class_means = [np.mean(perf_array[idxs]) for idxs in groups.values()]
        worst_mean = float(np.min(class_means))
        return size_balance, worst_mean

    toolbox.register("evaluate", eval_assignment)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=K-1, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)

    np.random.seed(seed)
    pop = toolbox.select([toolbox.individual() for _ in range(pop_size)], k=pop_size)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg_balance", lambda vs: np.mean([v[0] for v in vs]))
    stats.register("avg_worst", lambda vs: np.mean([v[1] for v in vs]))

    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size*2,
                                             cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                                             stats=stats, verbose=False)

    pareto = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    best = max(pareto, key=lambda ind: ind.fitness.values[1])

    # 5) Write cluster docs with explicit id
    cluster_docs = []
    for sid, grp in zip(student_ids, best):
        cluster_docs.append({
            "id": int(sid),
            "label": id_name_map.get(int(sid), ""),
            "cluster": int(grp)
        })

    return cluster_docs

