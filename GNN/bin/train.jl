using Pkg
Pkg.activate(@__DIR__)
using GNNProject, Graphs, GNNGraphs, BSON, Flux, Leiden, DataFrames


const OUTPUT_DIR = joinpath(@__DIR__, "..", "output")

const DATA_PATH = joinpath(@__DIR__, "..", "data/Student Survey - Jan.xlsx")
const MODEL_PATH = joinpath(OUTPUT_DIR, "models", "model.bson")
const TRAIN_RESULT_PATH = joinpath(OUTPUT_DIR, "models", "train_result.bson")
const EMBEDDING_PATH = joinpath(OUTPUT_DIR, "artifacts", "embeddings.bson")
const RESULTS_PATH = joinpath(OUTPUT_DIR, "artifacts", "results.bson")
const CLUSTERS_PATH = joinpath(OUTPUT_DIR, "artifacts", "clusters.bson")
const VIEWS_PATH = joinpath(OUTPUT_DIR, "artifacts", "views.bson")
const COMPOSITE_GRAPH_PATH = joinpath(OUTPUT_DIR, "artifacts", "composite_graph.bson")
const CLUSTERED_STUDENT_PATH = joinpath(OUTPUT_DIR, "artifacts", "clustered_students.bson")

graph_views, composite_graph, index_to_node = load_views_and_composite(DATA_PATH)

model = MultiViewGNN(
    # input dim
    size(graph_views[1].graph.ndata.topo, 1),
    # output dim
    size(composite_graph, 1)
)
opt = Flux.Adam(1e-3)
results = hyperparameter_search(
    model,
    graph_views,
    composite_graph,
    taus=[0.1f0, 0.5f0, 1.0f0],
    lambdas=[0.5f0, 1.0f0, 10.0f0],
    gammas=[0.01f0, 0.1f0],
    epochs=500,
    n_repeats=3
)

best_parameters = select_best_result(results)
trained_model = train_model(
    model,
    opt,
    graph_views,
    composite_graph;
    λ=best_parameters.λ,
    τ=best_parameters.τ,
    γ=best_parameters.γ,
    verbose=true,
    epochs=500
)

output = model(graph_views) |> cpu

norm_embeddings = Flux.normalise(output; dims=1)
knn = knn_graph(norm_embeddings, Int64(round(sqrt(size(output, 2)))))
clusters = leiden(adjacency_matrix(knn), "ngrb")
clustered_students = DataFrame(
    embedding_index=collect(vertices(composite_graph)),
    student_id=[
        index_to_node[id] for id in collect(vertices(composite_graph))
    ],
    cluster=clusters
)

BSON.@save TRAIN_RESULT_PATH trained_model = trained_model
BSON.@save MODEL_PATH model = cpu(trained_model.model)
BSON.@save RESULTS_PATH sweep_results = cpu(results)
BSON.@save EMBEDDING_PATH embeddings = cpu(output)
BSON.@save CLUSTERS_PATH clusters = cpu(clusters)
BSON.@save VIEWS_PATH views = cpu(graph_views)
BSON.@save COMPOSITE_GRAPH_PATH composite_graph = cpu(composite_graph)
BSON.@save CLUSTERED_STUDENT_PATH clustered_students = clustered_students
