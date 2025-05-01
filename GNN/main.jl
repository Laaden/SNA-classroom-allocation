using Pkg
Pkg.activate(".")
using Revise
using GNNProject

# ENV["CUDA_VISIBLE_DEVICES"] = "-1"

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

graph_views = [
    WeightedGraph(fr_mat, 0.4f0),
    WeightedGraph(inf_mat, 0.6f0),
    WeightedGraph(fd_mat, 0.8f0),
    WeightedGraph(mt_mat, 1f0),
    WeightedGraph(ad_mat, 0.9f0),
    WeightedGraph(ds_mat, -1.0f0),
    WeightedGraph(sc_mat, 0.1f0),
] |> gpu

composite_graph = reduce(
    (graph, (edge, weight)) -> add_edges(graph, (edge[1], edge[2], fill(weight, length(edge[1])))),
    zip(
        [edge_index(cpu(g.graph)) for g in graph_views],
        [g.weight[] for g in graph_views]
    ),
    init=GNNGraph()
)

# ~~ Setup Model ~~ #

model = MultiViewGNN(64, 64, size(composite_graph, 1)) |> gpu
opt = Flux.Adam(1e-3) |> gpu

# ~~ Train Model ~~ #

results = hyperparameter_search(
    model,
    graph_views,
    composite_graph,
    taus    = [0.5f0, 1f0],
    lambdas = [5.0f0, 10f0],
    gammas = [0.01f0, 0.1f0],
    epochs = 500,
    n_repeats = 3
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
    verbose = true,
    epochs = 500
)

# ~~ Model Output & Aggregation ~~ #
# This could technically end up as an algo as well

output = model(graph_views) |> cpu

# # ~~ Pass this off to community detection ~~ #

# # E.g. k-means
# # (I don't think we're using kmeans but it's illustrative)
# #
using Clustering
k = Int64(round(sqrt(size(output, 2))))
clusters = kmeans(Flux.normalise(output), k, maxiter=100)
intraview_cluster_rates = intra_cluster_rate(
    clusters.assignments,
    cpu(graph_views),
    ["friendship", "influence", "feedback", "more_time", "advice", "disrespect", "affiliation"]
)
composite_cluster_rates = intra_cluster_rate(clusters.assignments, composite_graph)


# # ~~ PSO ~~ #
# # do some PSO stuff at some point for class size & other node features

# using Plots, GraphPlot

# mod_search = best_parameters.logs.modularity
# silh = best_parameters.logs.silhouette
# acc = best_parameters.logs.accuracy
# loss = best_parameters.logs.loss

# epochs = collect(1:length(loss))
# plot(
#     epochs,
#     [
#         repeat(mod_search, inner = 10),
#         acc,
#         loss ./ 100,
#         repeat(silh, inner = 10)
#     ],
#     label=["Modularity" "Disc. Accuracy" "Loss / 100" "Silhouette"],
#     lw = 3
# )
# xlabel!("Epoch")
# ylabel!("Metric")
# title!("GNN Metrics over Epoch")
# plot!(legend=:outerbottom, legendcolumns=3, lw=10)
# yticks!(0:0.1:0.8)


# mod_train = trained_model.logs.modularity
# epochs = collect(1:length(mod_search)) .* 10
# plot(
#     epochs,
#     [mod_search, mod_train],
#     label=["Hyperparameter Modularity" "Train Modularity"],
#     lw = 3
# )
# xlabel!("Epoch")
# ylabel!("Metric")
# title!("Modularity over Epoch")
# plot!(legend=:outerbottom, legendcolumns=3, lw=10)
# best_epoch_search = epochs[argmax(mod_search)]
# best_epoch_train = epochs[argmax(mod_train)]
# vline!([best_epoch_search], label=false, color=:blue, linestyle=:dash)
# vline!([best_epoch_train], label=false, color=:orange, linestyle=:dash)

# using Colors

# colours = distinguishable_colors(maximum(clusters.assignments))
# group_colors = colours[clusters.assignments]

# gplot(
#     composite_graph,
#     nodefillc = group_colors
# )


function plot_graph(graph, labels::Vector{<:Real})
    colours = distinguishable_colors(maximum(labels))
    group_colors = colours[labels]
    edge_colours = [weight > 0 ? RGBA(1, 1, 1, 0.05) : RGBA(1, 0, 0, .5) for weight in get_edge_weight(graph)]
    layout = (args...) -> spring_layout(args...; C=30)
    gplot(
        graph,
        layout = layout,
        nodefillc = group_colors,
        linetype = "curve",
        edgestrokec = edge_colours,
        NODESIZE=0.03
    )
end

function plot_metric(results::Vector{<:TrainResult}, metric::Symbol)
    colours = cgrad(:viridis, length(results))
    final_scores = [getfield(r.logs, metric)[end] for r in results]
    sorted_indices = sortperm(final_scores, rev=true)
    p = plot()

    for (rank, idx) in enumerate(sorted_indices)
        r = results[idx]
        values = getfield(r.logs, metric)
        x_vals = 0:10:10*length(values)
        plot!(
         collect(1:length(values)).*10,
         values,
         label=false,
         alpha = 0.3 + 0.7 * (1 + idx - rank) / (idx - 1),
         lw = 3
        )
        if rank == 1
            annotate!(
                x_vals[end] - 50,
                values[end],
                text(
                    "λ $(r.λ), τ $(r.τ), γ $(r.γ)",
                    colours[idx],
                    8
                )
            )
        end
    end
    xlabel!("Epoch")
    ylabel!(String(metric))
    return p
end

function plot_metric(result1::TrainResult, result2::TrainResult, metric::Symbol)
    return plot_metric([result1, result2], metric)
end

p1 = plot_metric(results, :modularity)
p2 = plot_metric(results, :silhouette)
p3 = plot_metric(results, :conductance)
p4 = plot_metric(results, :accuracy)
p5 = plot_metric(results, :loss)


plot(p1, p2, p3, layout=(3, 1), link=:x)


plot_metric(trained_model, best_parameters, :conductance)
plot_graph(composite_graph, clusters.assignments)