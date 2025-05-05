using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GNNProject
using Plots
import CairoMakie as CM
using BSON: @load

# ~~ Setup ~~ #
OUTPUT_DIR = joinpath(@__DIR__, "..", "output")
@load joinpath(OUTPUT_DIR, "artifacts", "results.bson") sweep_results
@load joinpath(OUTPUT_DIR, "artifacts", "embeddings.bson") embeddings
@load joinpath(OUTPUT_DIR, "artifacts", "clusters.bson") clusters
@load joinpath(OUTPUT_DIR, "artifacts", "composite_graph.bson") composite_graph
@load joinpath(OUTPUT_DIR, "artifacts", "views.bson") views
@load joinpath(OUTPUT_DIR, "models", "train_result.bson") trained_model
logs_dir = joinpath(OUTPUT_DIR, "logs")

if !ispath(logs_dir) mkdir(logs_dir) end

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
        epochs=500
    ),
    ["friendship", "influence", "feedback", "more_time", "advice", "disrespect", "affiliation"]
)

savefig(accuracy_plot, joinpath(logs_dir, "accuracy.png"))
savefig(model_metrics, joinpath(logs_dir, "model_metrics.png"))
savefig(sweep_metrics, joinpath(logs_dir, "sweep.png"))
savefig(loss_composition_plot, joinpath(logs_dir, "loss_composition.png"))
savefig(embedding_plot, joinpath(logs_dir, "embeddings.png"))
# CM.save(joinpath(logs_dir, "network.png"), network)

summary_path = joinpath(logs_dir, "model_summary.md")
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

