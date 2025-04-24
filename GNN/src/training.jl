module ModelTraining
    include("loss.jl")
    include("model_evaluation.jl")
    using GraphNeuralNetworks, Graphs, Flux, CUDA, Statistics, Zygote
    using .Loss, .ModelEvaluation

    export train_model
    function train_model(model, projection_head, opt, discriminator, node_embedding, ps, views, graph; λ, τ, epochs = 300, verbose = false)
        # Tracking
        loss_log = []
        acc_log = []
        mod_log = []
        sil_log = []


        for epoch in 1:epochs
            loss_epoch = 0.0f0
            acc_epoch = 0.0f0
            mod_epoch = 0.0f0

            grads = Flux.gradient(ps) do
                for g in views
                    x = node_embedding(1:nv(g.graph))
                    loss, acc = contrastive_loss(
                        x,
                        model,
                        discriminator,
                        projection_head,
                        τ,
                        g
                    )
                    mod_loss = soft_modularity_loss(model, g, x)
                    acc_epoch += acc
                    mod_epoch += λ * mod_loss
                    loss_epoch += loss + λ * mod_loss
                end

                return loss_epoch
            end

            Flux.Optimise.update!(opt, ps, grads)

            if epochs % 10 == 0 && verbose == true
                contrast = loss_epoch - λ * mod_epoch
                @info "Epoch $(epoch) | Total Loss=$(round(loss_epoch, digits = 3)) " *
                  "| Contrast =$(round(contrast, digits = 3)) " *
                  "| Mod Loss =$(round(mod_epoch, digits = 4)) " *
                  "| Accuracy =$(round(acc_epoch/length(views), digits = 3))"
            end

            if epoch == epochs
                output = cpu(sum(abs(g.weight[1]) * model(g.graph, g.graph.ndata.x) for g in views))
                metrics = evaluate_embeddings(output, cpu(graph))
                push!(loss_log, loss_epoch)
                push!(acc_log, acc_epoch / length(views))
                push!(mod_log, metrics[:modularity])
                push!(sil_log, metrics[:silhouettes])
            end
        end

        return Dict(
            :model => model,
            :λ => λ,
            :τ => τ,
            :loss => loss_log,
            :acc => acc_log,
            :modularity => mod_log,
            :silhouette => sil_log
        )
    end

    export hyperparameter_search
    function hyperparameter_search(base_model, base_proj, base_disc, base_embed, views, graph; lambdas, taus, epochs)
        results = []
        for λ in lambdas, τ in taus
            @info "λ $(λ), τ $(τ)"
            model = gpu(deepcopy(base_model))
            proj = gpu(deepcopy(base_proj))
            disc = gpu(deepcopy(base_disc))
            embed = gpu(deepcopy(base_embed))
            opt = gpu(Flux.Adam(1e-3))
            ps = Flux.params(model, proj, disc, embed)
            result = train_model(model, proj, opt, disc, embed, ps, views, graph; λ=λ, τ=τ, epochs = epochs)
            push!(results, result)
        end

        return results
    end

end