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
        state = Flux.setup(opt, model)

        logs = TrainLog()
        graph = cpu(graph)

        # stop training early if modularity doesn't improve
        es =  Flux.early_stopping(
            () -> logs.modularity[end],
            15;
            distance = (a,b) -> b - a,
            min_dist = 1e-4
        )

        for epoch in 1:epochs
            loss_epoch = 0.0f0
            acc_epoch = 0.0f0
            mod_loss_epoch = 0.0f0
            contrast_epoch = 0.0f0
            balance_epoch = 0.0f0

            grads = Flux.gradient((model) -> begin
                total_loss, res = calculate_total_loss(model, views, τ, λ, γ)
                loss_epoch = total_loss
                mod_loss_epoch = res[:mod_loss]
                balance_epoch = res[:balance_loss]
                acc_epoch = res[:acc]
                contrast_epoch = res[:contrast_loss]
                return total_loss
            end, model)

            Flux.Optimise.update!(state, model, grads[1])

            push!(logs.loss.balance_loss, balance_epoch)
            push!(logs.loss.modularity_loss, mod_loss_epoch)
            push!(logs.loss.contrast_loss, contrast_epoch)
            push!(logs.loss.total_loss, loss_epoch)

            push!(logs.accuracy, acc_epoch / length(views))

            if epoch % 10 == 0 && verbose == true
                 @info "Epoch $epoch | " *
                 "Loss: $(round(loss_epoch, digits=3)) " *
                 "(C=$(round(contrast_epoch, digits=3)), " *
                 "M=$(round(mod_loss_epoch, digits=3)), " *
                 "B=$(round(balance_epoch, digits=3))) | " *
                 "Acc=$(round(acc_epoch / length(views), digits=3))"
            end

            if epoch % 10 == 0
                output = model(views)
                metrics = fast_evaluate_embeddings(cpu(output), graph)
                push!(logs.modularity, metrics[:modularity])
                push!(logs.silhouette, metrics[:silhouettes])
                push!(logs.conductance, metrics[:conductance])

                # we'll early stop if modularity hasn't improved in 150 epochs
                es() && break
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
