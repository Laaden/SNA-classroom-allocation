module Plotting
    using Plots, GNNGraphs, Graphs, GraphPlot, Colors
    using MultivariateStats, UMAP
    using ..Types

    export plot_network
    function plot_network(graph, labels::Vector{<:Real})
        colours = distinguishable_colors(maximum(labels))
        group_colors = colours[labels]
        edge_colours = [weight > 0 ? RGBA(1, 1, 1, 0.05) : RGBA(1, 0, 0, 0.5) for weight in get_edge_weight(graph)]
        layout = (args...) -> spring_layout(args...; C=30)
        gplot(
            graph,
            layout=layout,
            nodefillc=group_colors,
            linetype="curve",
            edgestrokec=edge_colours,
            NODESIZE=0.03
        )
    end

    export plot_metric
    function plot_metric(results::Vector{<:TrainResult}, metric::Symbol)
        colours = cgrad(:turbo)
        final_scores = [getfield(r.logs, metric)[end] for r in results]
        sorted_indices = sortperm(
            final_scores,
            rev= metric==:conductance ? false : true
        )
        p = plot()

        for (rank, idx) in enumerate(sorted_indices)
            r = results[idx]
            r_colour = get_colour_from_params(colours, r.λ, r.τ, r.γ)
            values = getfield(r.logs, metric)
            label = rank == 1 ? "λ $(r.λ), τ $(r.τ), γ $(r.γ)" : false
            plot!(
                collect(1:length(values)) .* 10,
                values,
                color = r_colour,
                label=label,
                alpha= rank == 1 ? 1 : 0.3,
                lw=3
            )
        end
        xlabel!("Epoch")
        ylabel!(String(metric))
        return p
    end

    export plot_loss_composition
    function plot_loss_composition(result::TrainResult)

        p = areaplot(
            1:length(result.logs.loss.total_loss),
            [result.logs.loss.contrast_loss result.logs.loss.balance_loss result.logs.loss.modularity_loss],
            fillalpha=[0.2, 0.3, 0.5],
            label=["Contrast" "Balance" "Modularity"],
            lw=0,
            framestyle=:semi,
            title = "Loss Composition",
            xlims=(0, Inf),
            ylims=(0, Inf)
        )
        xlabel!("Epoch")
        ylabel!("Loss")


        return p
    end

    export plot_embeddings
    function plot_embeddings(embeddings, assignments, method = :pca)

        if method == :pca
            pca_model = fit(PCA, embeddings; maxoutdim=2)
            X_pca = MultivariateStats.transform(pca_model, embeddings)
            x1 = X_pca[1, :]
            x2 = X_pca[2, :]
        elseif method == :umap
            proj = umap(embeddings, 2; n_neighbors=15)
            x1 = proj[1, :]
            x2 = proj[2, :]
        else
            error("Unsupported method: $method")
        end

        scatter(
            x1,
            x2,
            title=uppercase(string(method)) * " of Node Embeddings",
            xlabel="Component 1",
            ylabel="Component 2",
            group = assignments,
            labels = false
        )
    end

    function get_colour_from_params(gradient, λ, τ, γ)
        h = hash((λ, τ, γ)) % 10_000 / 10_000
        return gradient[h]
    end

end


