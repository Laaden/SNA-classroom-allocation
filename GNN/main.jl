using Pkg
Pkg.activate(".")
using Revise
using GNNProject

# ENV["CUDA_VISIBLE_DEVICES"] = "-1"

using Statistics, Graphs, Flux, GraphNeuralNetworks, Random, Zygote, DataFrames, CUDA, Leiden, Distances, LinearAlgebra

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
]

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
    taus    = [0.1f0, 0.5f0, 1f0],
    lambdas = [0.3f0, 0.5f0, 6.0f0, 10f0],
    epochs = 500,
    n_repeats = 3
)

# given community detection is the goal,
# modularity is our best metric for optimising the GNN
best_parameters = argmax(r -> begin
    m = maximum(r.logs.modularity)
    l = minimum(r.logs.loss)
    s = maximum(r.logs.silhouette)
    c = minimum(r.logs.conductance)
    return (m + s - l - c) / 4
    end,
    results
)

trained_model = train_model(
    model,
    opt,
    graph_views,
    composite_graph;
    λ=best_parameters.λ,
    τ=best_parameters.τ,
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
    (graph_views),
    ["friendship", "influence", "feedback", "more_time", "advice", "disrespect", "affiliation"]
)
composite_cluster_rates = intra_cluster_rate(
    clusters.assignments,
    composite_graph
)


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


# using Plots
# plot()
# for r in results
#     plot!(
#         10:10:10*length(r.logs.modularity),
#         r.logs.modularity,
#         label=false,
#         lw = 3
#     )
# end
# xlabel!("Epoch")
# ylabel!("Modularity")
# title!("Modularity curves with early stopping")



# function moving_average(x, w=3)
#     return [mean(x[max(1, i - w + 1):i]) for i in 1:length(x)]
# end

# plot()
# for r in results
#     ma_curve = moving_average(r.logs.modularity)
#     plot!(10:10:10*length(ma_curve), ma_curve, label=false)
# end
# xlabel!("Epoch")
# ylabel!("Modularity")
# title!("Smoothed Modularity Curves with Early Stopping")


# colors = cgrad(:viridis, length(results))  # color gradient
# final_modularities = [r.logs.modularity[end] for r in results]
# sorted_indices = sortperm(final_modularities, rev=true)  # highest first
# nudge = [-0.0, 0.00, -0.02]

# plot()
# for (rank, idx) in enumerate(sorted_indices)
#     r = results[idx]
#     ma_curve = moving_average(r.logs.modularity)
#     x_vals = 10:10:10*length(ma_curve)
#     plot!(
#         10:10:10*length(ma_curve),
#          ma_curve,
#          color=colors[idx * 10],
#          alpha = 0.3 + 0.7 *  (idx - rank) / (idx - 1),
#          label=false,
#          lw = 3
#     )
#     if rank <= 3
#         annotate!(
#             x_vals[end] - 50,    # X position (slightly before final epoch)
#             ma_curve[end] + nudge[rank],       # Y position (final modularity value)
#             text("λ=$(r.λ), τ=$(r.τ)", :black, 8)  # Text label, color, size
#         )
#     end
# end
# xlabel!("Epoch")
# ylabel!("Modularity")
# title!("Modularity Colored by Final Performance")
