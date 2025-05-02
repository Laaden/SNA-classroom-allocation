using Pkg
Pkg.activate(@__DIR__)
using Revise, BSON
using GNNProject

ENV["CUDA_VISIBLE_DEVICES"] = "-1"

using Statistics
using Flux
using GraphNeuralNetworks
using Random
using Zygote
using DataFrames
using CUDA
using Leiden
using Distances
using LinearAlgebra
using Graphs, GraphPlot

# ~~ Read in data and setup data structures ~~ #

xlsx_file = "./data/Student Survey - Jan.xlsx"

fr_mat, inf_mat, fd_mat, mt_mat, ad_mat, ds_mat, sc_mat = create_adjacency_matrix(
	matrix_from_sheet(xlsx_file, "net_0_Friends"),
    matrix_from_sheet(xlsx_file, "net_1_Influential"),
    matrix_from_sheet(xlsx_file, "net_2_Feedback"),
    matrix_from_sheet(xlsx_file, "net_3_MoreTime"),
    matrix_from_sheet(xlsx_file, "net_4_Advice"),
    matrix_from_sheet(xlsx_file, "net_5_Disrespect"),
    matrix_from_sheet(xlsx_file, "net_affiliation_0_SchoolActivit"),
)

graph_views = gpu.([
    WeightedGraph(fr_mat, 0.4f0),
    WeightedGraph(inf_mat, 0.6f0),
    WeightedGraph(fd_mat, 0.8f0),
    WeightedGraph(mt_mat, 1f0),
    WeightedGraph(ad_mat, 0.9f0),
    WeightedGraph(ds_mat, -1.0f0),
    WeightedGraph(sc_mat, 0.1f0),
])

composite_graph = reduce(
    (graph, (edge, weight)) -> add_edges(graph, (edge[1], edge[2], fill(weight, length(edge[1])))),
    zip(
        [edge_index(cpu(g.graph)) for g in graph_views],
        [g.weight[] for g in graph_views]
    ),
    init=GNNGraph()
)

# ~~ Setup Model ~~ #


model_path = joinpath(@__DIR__, "models", "model.bson")

if isfile(model_path)
    BSON.@load model_path model
    model = gpu(model)
else
    model = MultiViewGNN(64, 64, size(composite_graph, 1)) |> gpu
    opt = Flux.Adam(1e-3) |> gpu

    # ~~ Train Model ~~ #

    results = hyperparameter_search(
        model,
        graph_views,
        composite_graph,
        taus=[0.5f0, 1.0f0],
        lambdas=[5.0f0, 10.0f0],
        gammas=[0.01f0, 0.1f0],
        epochs=500,
        n_repeats=3
    )

    # given community detection is the goal,
    # modularity is our best metric for optimising the GNN
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
    BSON.@save "model.bson" model = cpu(model)
end


# ~~ Model Output & Aggregation ~~ #
# This could technically end up as an algo as well

output = model(graph_views) |> cpu

# # ~~ Pass this off to community detection ~~ #

norm_embeddings = Flux.normalise(output; dims=1)
knn = knn_graph(norm_embeddings, Int64(round(sqrt(size(output, 2)))))
clusters = leiden(adjacency_matrix(knn), "ngrb")

mod_eval = model_summary(
    output,
    clusters,
    composite_graph,
    cpu(graph_views),
    (
        λ=trained_model.λ,
        τ=trained_model.τ,
        γ=trained_model.γ,
        epochs = 500
    ),
    ["friendship", "influence", "feedback", "more_time", "advice", "disrespect", "affiliation"]
)

# # ~~ PSO ~~ #
# # do some PSO stuff at some point for class size & other node features


using Plots

accuracy = plot_metric(results, :accuracy)

plot(
    plot_metric(results, :modularity),
    plot_metric(results, :silhouette),
    plot_metric(results, :conductance),
    layout=(3, 1),
    link=:x,
    xlabel="Epoch"
)

plot(
    plot_metric([trained_model], :modularity),
    plot_metric([trained_model], :silhouette),
    plot_metric([trained_model], :conductance),
    layout=(3, 1),
    xlabel="Epoch"
)


plot_loss_composition(best_parameters)
plot_loss_composition(trained_model)


plot(
    plot_embeddings(output, clusters, :pca),
    plot_embeddings(output, clusters, :umap),
    layout=(1, 2)
)
