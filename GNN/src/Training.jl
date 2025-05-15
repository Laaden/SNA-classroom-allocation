 module ModelTraining
    using GraphNeuralNetworks, Graphs, Flux, Statistics, Zygote, Random
    using ..Loss, ..ModelEvaluation, ..Types

    # This GNN uses a multi-view representational learning approach,
    # wherein discrete views are passed through the same
    # gradient tape. We use a multi-objective approach for the training:
    #   1. Modified DGI contrastive loss
    #       - Bilinear discriminator with negative sampling
    #       - Projection head improves discrimination
    #       - Repulsive graphs are treated as negative signals
    #   2. Modified Soft modularity optimisation
    #       - Repulsive graphs reduce soft modularity by having their adjacency mats negated,
    #         encouraging negative views to be pushed into different clusters
    #
    # Weights are intentionally NOT applied to the training outputs.
    # This is for a few reasons
    #   1.  GraphSage doesn't natively use edge weights. Technically there are workarounds, but:
    #   2. Real-time weight modification at inference was a hard requirement.
    #
    # Weights *are* technically used in the training, for their polarity only.
    # The magnitude however, is left to post-hoc weighted embedding aggregation.
    #
    # It also uses a projection head during the contrastive loss
    # (inspired by SimCLR), which transforms embeddings into a separate space
    # optimised for DGI contrastive learning.
    export train_model
    function train_model(model::MultiViewGNN, opt, views::Vector{WeightedGraph}, graph::GNNGraph; λ::Float32, τ::Float32, γ::Float32, epochs::Int64=300, verbose::Bool=false)::TrainResult
        init_uncertainty!(model, views; τ=τ)

        state = Flux.setup(opt, model)
        logs = TrainLog()
        graph = cpu(graph)

        # stop training early if modularity/loss doesn't improve
        es_mod = Flux.early_stopping(() -> logs.loss.modularity_loss[end], 30; min_dist=1e-3)
        es_total = Flux.early_stopping(() -> logs.loss.total_loss[end], 30; min_dist=1e-3)

        for epoch in 1:epochs

            grads = Flux.gradient((model) -> begin
                loss, _ = multitask_loss(model, views)
                return loss
            end, model)

            Flux.Optimise.update!(state, model, grads[1])

            Lc, Lm, Lb, Acc = compute_task_losses(model, views, 1.0f0)
            loss_c = 0.5f0 * exp(-2f0*model.logσ_c[1]) * Lc + model.logσ_c[1]
		    loss_m = 0.5f0 * exp(-2f0*model.logσ_m[1]) * Lm + model.logσ_m[1]
		    loss_b = 0.5f0 * exp(-2f0*model.logσ_b[1]) * Lb + model.logσ_b[1]
            total_loss = loss_c + loss_m + loss_b
            push!(logs.loss.contrast_loss, loss_c)
            push!(logs.loss.modularity_loss, loss_m)
            push!(logs.loss.balance_loss, loss_b)
            push!(logs.loss.total_loss, total_loss)
            push!(logs.accuracy, Acc / length(views))

            if epoch % 10 == 0
                output = model(views)
                metrics = evaluate_embeddings(cpu(output), graph)
                push!(logs.modularity, metrics[:modularity])
                push!(logs.silhouette, metrics[:silhouettes])
                push!(logs.conductance, metrics[:conductance])
                push!(logs.embeddings, output)

                if verbose == true
                    w_c = 0.5f0 * exp(-2f0*model.logσ_c[1])
                    w_m = 0.5f0 * exp(-2f0*model.logσ_m[1])
                    w_b = 0.5f0 * exp(-2f0*model.logσ_b[1])

                    @info "Epoch $epoch | " *
                    "Loss: $(round(total_loss, digits=3)) " *
                    "Lc=$(round(loss_c, digits=3)), " *
                    "Lm=$(round(loss_m, digits=3)), " *
                    "Lb=$(round(loss_b, digits=3))) | " *
                    "Acc=$(round(Acc / length(views), digits=3))) | \n" *
                    "\t→ Modularity = $(round(metrics[:modularity], digits=3)), " *
                    "Silhouette = $(round(metrics[:silhouettes], digits=3)), " *
                    "Conductance = $(round(metrics[:conductance], digits=3)) \n" *
                    "\t→ task-weights: w_c=$(round(w_c, digits = 3)), w_m=$(round(w_m, digits = 3)), w_b=$(round(w_b, digits = 3))"


                end


                if false #es_mod()
                    @info "Early modularity stopping triggered at epoch: $epoch"
                    break
                elseif false #es_total()
                    @info "Early loss stopping triggered at epoch: $epoch"
                    break
                end
            end
        end

        return TrainResult(model, λ, τ, γ, logs)
    end

    export hyperparameter_search
    function hyperparameter_search(base_model::MultiViewGNN, views::Vector{WeightedGraph}, graph::GNNGraph; lambdas, taus, gammas, epochs, n_repeats)::Vector{TrainResult}
        results::Vector{TrainResult} = TrainResult[]
        configs = collect(Iterators.product(lambdas, taus, gammas))

        # todo, see if this can be threaded or parallel-processed
        # potentially difficult when working with GPUs
        for  (i, (λ, τ, γ)) in enumerate(configs)
            config_results::Vector{TrainResult} = TrainResult[]
            @info "Running config $i of $(length(configs)) | λ = $λ, τ = $τ, γ = $γ"
            for n in 1:n_repeats
                Random.seed!(1000 * i + n)
                # todo send these to gpu if needed
                # removed for package compilation
                model = deepcopy(base_model)
                opt = Flux.Adam(1e-3)
                result = train_model(model, opt, views, graph; λ=λ, τ=τ, γ=γ, epochs=epochs)
                push!(config_results, result)
            end
            best = argmax(r -> mean(r.logs.modularity), config_results)
            push!(results, best)
        end

        for (i, r) in enumerate(results)
            best_mod = maximum(r.logs.modularity)
            @info "Run $i | λ=$(r.λ), τ=$(r.τ), γ=$(r.γ) → max modularity = $(round(best_mod, digits=3))"
        end

        return results
    end

    export select_best_result
    function select_best_result(results::Vector{TrainResult})
        return argmax(r -> begin
                m = (maximum(r.logs.modularity))
                l = (minimum(r.logs.loss.total_loss))
                s = (maximum(r.logs.silhouette))
                c = (minimum(r.logs.conductance))
                return (m + s - l - c) / 4
            end,
            results
        )
    end

    function minmax(x)
        return (x .- minimum(x)) ./ (maximum(x) - minimum(x) + eps())
    end

end
