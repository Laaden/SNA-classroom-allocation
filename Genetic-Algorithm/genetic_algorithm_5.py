import os
import pandas as pd
import numpy as np
from collections import Counter
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from pymongo import MongoClient

# ——— 1) Load raw data and cluster labels ———
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://3.105.47.11:27017")
client = MongoClient(MONGO_URI)
db = client["sna_database"]

# Raw survey/edge data
df_raw = pd.DataFrame(list(db.sna_student_raw.find({}, {"_id": 0})))
if "Participant-ID" in df_raw.columns:
    df_raw["Participant-ID"] = pd.to_numeric(
        df_raw["Participant-ID"], errors="coerce"
    ).astype(int)

# Leiden cluster assignments
df_clusters = pd.read_csv("clustered_students.csv")
df_clusters["student_id"] = pd.to_numeric(
    df_clusters["student_id"], errors="coerce"
).astype(int)

# ——— 2) Merge in academic performance ———
perf_field = "Perc_Academic July"
if perf_field not in df_raw.columns:
    raise KeyError(f"Expected performance field '{perf_field}' not found")

df_raw[perf_field] = pd.to_numeric(df_raw[perf_field], errors="coerce")

df = (
    df_clusters[["student_id"]]
      .merge(
          df_raw[["Participant-ID", perf_field]]
            .rename(columns={
                "Participant-ID": "student_id",
                perf_field:        "Perc_Academic"
            }),
          on="student_id",
          how="left"
      )
)

# Fill missing scores with cohort mean
df["Perc_Academic"].fillna(df["Perc_Academic"].mean(), inplace=True)

# ——— 3) Prepare GA inputs ———
student_ids = df["student_id"].tolist()
perf_array  = df["Perc_Academic"].values.astype(float)
N = len(student_ids)

# ——— 4) Define multi-objective GA (size balance ↓, worst-class mean ↑) ———
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_group", np.random.randint, 0, N)
toolbox.register("individual",
                 tools.initRepeat,
                 creator.Individual,
                 toolbox.attr_group,
                 n=N)

def eval_assignment(individual):
    # 1) Size balance: standard deviation of group sizes
    sizes = np.array(list(Counter(individual).values()), dtype=float)
    size_balance = float(np.std(sizes))
    # 2) Worst-class mean: lowest average academic score among groups
    groups = {}
    for idx, g in enumerate(individual):
        groups.setdefault(g, []).append(idx)
    class_means = [np.mean(perf_array[idxs]) for idxs in groups.values() if idxs]
    worst_mean = float(np.min(class_means))
    return size_balance, worst_mean

toolbox.register("evaluate", eval_assignment)
toolbox.register("mate",   tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt,
                 low=0, up=N-1, indpb=0.05)
toolbox.register("select", tools.selNSGA2)

# ——— 5) Run GA & save results ———
def main(seed=42):
    np.random.seed(seed)
    # Initial population
    pop = [toolbox.individual() for _ in range(200)]
    pop = tools.selNSGA2(pop, k=len(pop))

    # Track statistics
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg_size_balance", lambda vals: np.mean([f[0] for f in vals]))
    stats.register("avg_worst_mean",    lambda vals: np.mean([f[1] for f in vals]))

    # Mu+Lambda evolution
    pop, logbook = algorithms.eaMuPlusLambda(
        population=pop,
        toolbox=toolbox,
        mu=200,
        lambda_=400,
        cxpb=0.6,
        mutpb=0.3,
        ngen=50,
        stats=stats,
        verbose=True
    )

    # Save progress
    pd.DataFrame(logbook).to_csv("ga_progress.csv", index=False)

    # Select best Pareto solution (maximising worst-class mean)
    pareto = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
    best   = max(pareto, key=lambda ind: ind.fitness.values[1])

    # Write final assignment
    pd.DataFrame({
        "student_id":  student_ids,
        "group_label": best
    }).to_csv("ga_allocation_solution.csv", index=False)

    # Plot improvement of worst-class mean over generations
    df_log = pd.DataFrame(logbook)
    plt.figure(figsize=(6,4))
    plt.plot(df_log["gen"], df_log["avg_worst_mean"], label="Avg Worst-Class Mean")
    plt.xlabel("Generation")
    plt.ylabel("Worst-Class Mean")
    plt.title("Improvement of Worst-Class Mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig("academic_improvement.png")

if __name__ == "__main__":
    main()