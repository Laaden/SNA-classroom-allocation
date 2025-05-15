using Pkg
Pkg.activate(@__DIR__)
using GNNProject, Graphs, GNNGraphs, BSON, Flux, Leiden, DataFrames, Statistics


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

trained_model = train_model(
    model,
    opt,
    graph_views,
    composite_graph;
    λ=1.0f0,#best_parameters.λ,
    τ=1.0f0, #best_parameters.τ,
    γ=1.0f0,#best_parameters.γ,
    verbose=true,
    epochs=2000
)

output = model(graph_views)

norm_embeddings = Flux.normalise(output; dims=1)
avg_deg = mean([2 * ne(v.graph) / nv(v.graph) for v in graph_views if v.weight[] > 0])
k = min(round(Int, avg_deg), 15)
knn = knn_graph(norm_embeddings, k)
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
# BSON.@save RESULTS_PATH sweep_results = cpu(results)
BSON.@save EMBEDDING_PATH embeddings = cpu(output)
BSON.@save CLUSTERS_PATH clusters = cpu(clusters)
BSON.@save VIEWS_PATH views = cpu(graph_views)
BSON.@save COMPOSITE_GRAPH_PATH composite_graph = cpu(composite_graph)
BSON.@save CLUSTERED_STUDENT_PATH clustered_students = clustered_students
