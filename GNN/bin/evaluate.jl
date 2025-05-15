using Pkg
Pkg.activate(@__DIR__)


using Plots, GraphMakie, MultivariateStats, NetworkLayout, UMAP
import CairoMakie as CM
using BSON: @load

using GNNProject
include("Plotting.jl")

# ~~ Setup ~~ #
const OUTPUT_DIR = joinpath(@__DIR__, "..", "output")
const LOGS_DIR = joinpath(OUTPUT_DIR, "logs")

# @load joinpath(OUTPUT_DIR, "artifacts", "results.bson") sweep_results
@load joinpath(OUTPUT_DIR, "artifacts", "embeddings.bson") embeddings
@load joinpath(OUTPUT_DIR, "artifacts", "clusters.bson") clusters
@load joinpath(OUTPUT_DIR, "artifacts", "composite_graph.bson") composite_graph
@load joinpath(OUTPUT_DIR, "artifacts", "views.bson") views
@load joinpath(OUTPUT_DIR, "models", "train_result.bson") trained_model

if !ispath(LOGS_DIR)
    mkdir(LOGS_DIR)
end

# ~~ Global Metric Plots ~~ #
accuracy_plot = Plotting.plot_metric([trained_model], :accuracy)

# sweep_metrics = plot(
#     Plotting.plot_metric(sweep_results, :modularity),
#     Plotting.plot_metric(sweep_results, :silhouette),
#     Plotting.plot_metric(sweep_results, :conductance),
#     layout=(3, 1),
#     link=:x,
#     xlabel="Epoch"
# )

# ~~ Per-model Plot ~~ #
loss_composition_plot = Plotting.plot_loss_composition(trained_model)

embedding_plot = plot(
    Plotting.plot_embeddings(embeddings, clusters, :pca),
    Plotting.plot_embeddings(embeddings, clusters, :umap),
    layout=(1, 2)
)

model_metrics = plot(
    Plotting.plot_metric([trained_model], :modularity),
    Plotting.plot_metric([trained_model], :silhouette),
    Plotting.plot_metric([trained_model], :conductance),
    layout=(3, 1),
    link=:x,
    xlabel="Epoch"
)

# network = plot_network(
#     composite_graph,
#     clusters
# )

model_evaluation = model_summary(
    embeddings,
    clusters,
    composite_graph,
    views,
    (
        λ=trained_model.λ,
        τ=trained_model.τ,
        γ=trained_model.γ,
        epochs=1000
    ),
    ["friendship", "influence", "feedback", "more_time", "advice", "disrespect", "affiliation"]
)

savefig(accuracy_plot, joinpath(LOGS_DIR, "accuracy.png"))
savefig(model_metrics, joinpath(LOGS_DIR, "model_metrics.png"))
# savefig(sweep_metrics, joinpath(LOGS_DIR, "sweep.png"))
savefig(loss_composition_plot, joinpath(LOGS_DIR, "loss_composition.png"))
savefig(embedding_plot, joinpath(LOGS_DIR, "embeddings.png"))
# CM.save(joinpath(LOGS_DIR, "network.png"), network)

summary_path = joinpath(LOGS_DIR, "model_summary.md")
open(summary_path, "w") do io
    println(io, "# Baseline Graph Summary")
    println(io, "\n## Cluster Quality\n")
    for (k, v) in pairs(model_evaluation[:baseline_metrics][:embedding])
        if v isa AbstractDict
            println(io, "- $k:")
            for (k2, v2) in pairs(v)
                println(io, "  - $k2 = $v2")
            end
        else
            println(io, "- $k = $v")
        end
    end

    println(io, "\n## Intra-cluster Composite Rate\n")

    println(io, "- Negative: ", model_evaluation[:baseline_metrics][:clustering][:intra_cluster][:composite][:negative_intra])
    println(io, "- Positive: ", model_evaluation[:baseline_metrics][:clustering][:intra_cluster][:composite][:positive_intra])

    println(io, "\n## Per-view Cluster Rate\n")
    for (k, v) in pairs(model_evaluation[:baseline_metrics][:clustering][:intra_cluster][:per_view])
        println(io, "- $k = $(v[:intra])")
    end

    println(io, "\n# Model Summary")
    println(io, "\n## Hyperparameters\n")
    for (k, v) in pairs(model_evaluation[:model][:parameters])
        println(io, "- $k = $v")
    end

    println(io, "\n## Cluster Quality\n")
    for (k, v) in pairs(model_evaluation[:metrics][:embedding])
        if v isa AbstractDict
            println(io, "- $k:")
            for (k2, v2) in pairs(v)
                println(io, "  - $k2 = $v2")
            end
        else
            println(io, "- $k = $v")
        end
    end

    println(io, "\n## Intra-cluster Composite Rate\n")

    println(io, "- Negative: ", model_evaluation[:metrics][:clustering][:intra_cluster][:composite][:negative_intra])
    println(io, "- Positive: ", model_evaluation[:metrics][:clustering][:intra_cluster][:composite][:positive_intra])

    println(io, "\n## Per-view Cluster Rate\n")
    for (k, v) in pairs(model_evaluation[:metrics][:clustering][:intra_cluster][:per_view])
        println(io, "- $k = $(v[:intra])")
    end
end

