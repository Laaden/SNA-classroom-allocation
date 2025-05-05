using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GNNProject
using Plots
using BSON: @load

# ~~ Setup ~~ #
output_dir = joinpath(@__DIR__, "..", "output")
logs_dir = joinpath(output_dir, "logs")

@load joinpath(output_dir, "artifacts", "results.bson") sweep_results
@load joinpath(output_dir, "artifacts", "embeddings.bson") embeddings
@load joinpath(output_dir, "artifacts", "clusters.bson") clusters
@load joinpath(output_dir, "artifacts", "composite_graph.bson") composite_graph
@load joinpath(output_dir, "artifacts", "views.bson") views
@load joinpath(output_dir, "models", "train_result.bson") trained_model

# ~~ Global Metric Plots ~~ #
accuracy_plot = plot_metric(sweep_results, :accuracy)

sweep_metrics = plot(
    plot_metric(sweep_results, :modularity),
    plot_metric(sweep_results, :silhouette),
    plot_metric(sweep_results, :conductance),
    layout=(3, 1),
    link=:x,
    xlabel="Epoch"
)

# ~~ Per-model Plot ~~ #
loss_composition_plot = plot_loss_composition(trained_model)

embedding_plot = plot(
    plot_embeddings(embeddings, clusters, :pca),
    plot_embeddings(embeddings, clusters, :umap),
    layout=(1, 2)
)

model_metrics = plot(
    plot_metric([trained_model], :modularity),
    plot_metric([trained_model], :silhouette),
    plot_metric([trained_model], :conductance),
    layout=(3, 1),
    link=:x,
    xlabel="Epoch"
)


model_evaluation = model_summary(
    embeddings,
    clusters,
    composite_graph,
    views,
    (
        λ=trained_model.λ,
        τ=trained_model.τ,
        γ=trained_model.γ,
        epochs=500
    ),
    ["friendship", "influence", "feedback", "more_time", "advice", "disrespect", "affiliation"]
)

savefig(accuracy_plot, joinpath(logs_dir, "accuracy.png"))
savefig(model_metrics, joinpath(logs_dir, "model_metrics.png"))
savefig(sweep_metrics, joinpath(logs_dir, "sweep.png"))
savefig(loss_composition_plot, joinpath(logs_dir, "loss_composition.png"))
savefig(embedding_plot, joinpath(logs_dir, "embeddings.png"))

summary_path = joinpath(logs_dir, "model_summary.md")
open(summary_path, "w") do io
    println(io, "# Model Summary")
    println(io, "\n## Hyperparameters\n")
    for (k, v) in pairs(model_evaluation[:model][:parameters])
        println(io, "- $k = $v")
    end

    println(io, "\n## Embedding Quality\n")
    for (k, v) in pairs(model_evaluation[:metrics][:embedding])
        println(io, "- $k = $v")
    end

    println(io, "\n## Intra-cluster Composite Rate\n")

    println(io, "- Negative: ", model_evaluation[:metrics][:clustering][:intra_cluster][:composite][:negative_intra])
    println(io, "- Positive: ", model_evaluation[:metrics][:clustering][:intra_cluster][:composite][:positive_intra])
end
