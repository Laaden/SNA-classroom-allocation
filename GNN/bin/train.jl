using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using GNNProject, GNNGraphs, BSON, Flux, Leiden


output_dir = joinpath(@__DIR__, "..", "output")
xlsx_file = joinpath(@__DIR__, "..", "data/Student Survey - Jan.xlsx")
model_path = joinpath(output_dir, "models", "model.bson")
train_result_path = joinpath(output_dir, "models", "train_result.bson")
embedding_path = joinpath(output_dir, "artifacts", "embeddings.bson")
results_path = joinpath(output_dir, "artifacts", "results.bson")
clusters_path = joinpath(output_dir, "artifacts", "clusters.bson")
views_path = joinpath(output_dir, "artifacts", "views.bson")
composite_graph_path = joinpath(output_dir, "artifacts", "composite_graph.bson")

graph_views, composite_graph = load_views_and_composite(xlsx_file)
model = MultiViewGNN(64, 64, size(composite_graph, 1)) |> gpu
opt = Flux.Adam(1e-3) |> gpu
results = hyperparameter_search(
    model,
    graph_views,
    composite_graph,
    taus=[0.1f0, 0.5f0, 1.0f0],
    lambdas=[0.5f0, 1.0f0, 5.0f0, 10.0f0],
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

BSON.@save train_result_path trained_model = trained_model
BSON.@save model_path model = cpu(trained_model.model)
BSON.@save results_path sweep_results = cpu(results)
BSON.@save embedding_path embeddings = cpu(output)
BSON.@save clusters_path clusters = cpu(clusters)
BSON.@save views_path views = cpu(graph_views)
BSON.@save composite_graph_path composite_graph = cpu(composite_graph)
