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
    # Connect & name map
    if mongo_uri is None:
        mongo_uri = os.environ.get("MONGO_URI", "mongodb://3.105.47.11:27017")
    client = MongoClient(mongo_uri)
    db = client[db_name]

    # Load class size weight slider
    weights = db.sna_weights.find_one({}, {"_id": 0})
    csw_pct = float(weights.get("classSize", 50))
    csw = csw_pct / 100.0 

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
        clusters_json = list(db.gnn_results.find({}))
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
        raw_balance = float(np.std(sizes))
        size_balance = raw_balance * csw
        groups = {}
        for idx, g in enumerate(ind): groups.setdefault(g, []).append(idx)
        class_means = [np.mean(perf_array[idxs]) for idxs in groups.values()]
        worst_mean = float(np.min(class_means))
        # proportion changed from gnn to ga
        gnn_diff = sum(a != b for a, b in zip(ind, gnn_labels)) / len(ind)
        gnn_penalty = 1.0 * gnn_diff

        return size_balance + gnn_penalty, worst_mean

    toolbox.register("evaluate", eval_assignment)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=K-1, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)

    np.random.seed(seed)

    # use gnn results as base population
    gnn_labels = df_clusters.set_index("student_id").loc[student_ids]["cluster"].tolist()
    initial_population = [creator.Individual(list(gnn_labels)) for _ in range(pop_size)]
    pop = toolbox.select(initial_population, k=pop_size)

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

    original = gnn_labels
    final = best
    changes = sum(1 for a, b in zip(original, final) if a != b)
    percent_changed = changes / len(final) * 100

    print(f"GA changed {changes} students ({percent_changed:.2f}%) from the GNN result")

    return cluster_docs
