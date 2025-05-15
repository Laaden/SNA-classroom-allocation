module Types
    using GraphNeuralNetworks, Graphs, Flux, StatsBase, Base.Threads, Zygote

    const VIEW_NAMES = [
        "friendship", "influence", "feedback", "more_time",
        "advice", "disrespect", "affiliation"
    ]

    const VIEW_INDEX = Dict(name => i for (i, name) in enumerate(VIEW_NAMES))
    const NUM_VIEWS = length(VIEW_NAMES)


    function safe_zscore(x::AbstractVector)
        x_std = std(x)
        return x_std ≈ 0 ? fill(0.0f0, length(x)) : Float32.((x .- mean(x)) ./ (x_std + eps()))
    end

    function normalise_across_graph(x::AbstractMatrix{<:Real})
        return (x .- mean(x; dims=2)) ./ (std(x; dims=2) .+ eps())
    end


    export WeightedGraph
    struct WeightedGraph
        graph::GNNGraph
        weight::Ref{Float32}
        adjacency_matrix::AbstractMatrix{<:Real}
        view_type::String

        function WeightedGraph(adj_mat::AbstractMatrix{<:Integer}, weight::Float32, view_name::String)
            g = GNNGraph(adj_mat)
            # Topological node features were chosen using
            # PCA to see the explanatory power of each
            # feature.
            # i.e. fit(PCA, features; maxdim = n_features)
            #
            topo_features = vcat(
                get_topological_features(g),
                get_egonet_topological_features(g)
            )
            replace!(topo_features, NaN => 0.0f0)
            replace!(topo_features, Inf => 0.0f0)
            g.ndata.topo = topo_features

            return new(g, Ref(weight), Float32.(adj_mat), view_name)
        end

        function WeightedGraph(graph::GNNGraph, weight::Ref{Float32}, adjacency::AbstractMatrix{<:Real}, view_name::String)
            return new(graph, weight, adjacency, view_name)
        end
    end

    function get_topological_features(g::AbstractGraph)
        Float32.(hcat(
            local_clustering_coefficient(g) |> safe_zscore,
            indegree_centrality(g) |> safe_zscore,
            outdegree_centrality(g) |> safe_zscore,
            # betweenness_centrality(g) |> safe_zscore,
            pagerank(g) |> safe_zscore,
            # triangles(g) |> safe_zscore,
            # eigenvector_centrality(g) |> safe_zscore
        ))'
    end

    function get_egonet_topological_features(g)
        egonets = [induced_subgraph(g, neighborhood(g, v, 1)) for v in 1:nv(g)]

        ego_node_features = [
            begin
                nodes = neighborhood(g, ego_idx, 1)
                idx = findfirst(==(ego_idx), nodes)


                degs = degree(egonet)
                neighbour_degrees = deleteat!(copy(degs), idx)

                features = [
                    ne(egonet),
                    degs[idx],
                    density(egonet),
                    local_clustering_coefficient(egonet)[idx],
                    mean(neighbour_degrees)
                ]
            end for (ego_idx, egonet) in enumerate(egonets)
        ]

        egonet_mtx = hcat(ego_node_features...)
        replace!(egonet_mtx, NaN => 0.0f0)
        replace!(egonet_mtx, Inf => 0.0f0)
        egonet_mtx = normalise_across_graph(egonet_mtx)

        return Float32.(egonet_mtx)
    end


    export sample_weighted_subgraph
    function sample_weighted_subgraph(wg::WeightedGraph, node_frac = 0.8f0)
        nodes = (rand(Bool, nv(wg.graph)) .< node_frac) |>
            findall
        subg = induced_subgraph(wg.graph, nodes)
        adj = Matrix(adjacency_matrix(subg, Int64))
        # check check this, it might be brittle to the secondary constructor not using ref?
        WeightedGraph(adj, wg.weight[], wg.view_type)
    end

    Zygote.@nograd sample_weighted_subgraph


    # Adapt.@adapt_structure WeightedGraph

    struct LossLogs
        total_loss::Vector{Float32}
        balance_loss::Vector{Float32}
        contrast_loss::Vector{Float32}
        modularity_loss::Vector{Float32}
        function LossLogs(
            total::Vector{Float32} = Float32[],
            balance::Vector{Float32} = Float32[],
            contrast::Vector{Float32} = Float32[],
            modularity::Vector{Float32} = Float32[]
        )
            return new(total, balance, contrast, modularity)
        end
    end


    export TrainLog
    struct TrainLog
        loss::LossLogs
        accuracy::Vector{Float32}
        modularity::Vector{Float32}
        silhouette::Vector{Float32}
        conductance::Vector{Float32}
        function TrainLog(
            loss::LossLogs = LossLogs(),
            accuracy::Vector{Float32} = Float32[],
            modularity::Vector{Float32} = Float32[],
            silhouette::Vector{Float32} = Float32[],
            conductance::Vector{Float32} = Float32[]
        )
            return new(loss, accuracy, modularity, silhouette, conductance)
        end
    end

    export TrainResult
    struct TrainResult{T}
        model::T
        λ::Float32
        τ::Float32
        γ::Float32
        logs::TrainLog
    end

    export MultiViewGNN
    struct MultiViewGNN
        encoder::Chain
        # 2-layer GraphSage GNN
        layers::NamedTuple
        # projection head taken from SimCLR,
        # seems to help contrastive loss differentiate better
        # it's only used during training, we don't apply it on the final foreward pass
        projection_head::Chain
        # Used for DGI contrastive loss
        discriminator::Flux.Bilinear
        view_embeddings::Flux.Embedding

        logσ_c::Vector{Float32}
        logσ_m::Vector{Float32}
        logσ_b::Vector{Float32}
    end

    Flux.@layer MultiViewGNN

    # Outer constructor for the MultiViewGNN
    function MultiViewGNN(input_dim::Int64, output_dim::Int64)
        hidden_dim = 32

         # per-view learned embeddings
        embeddings = Flux.Embedding(NUM_VIEWS => input_dim)

        input_dim = input_dim * 2

        encoder = Chain(
            Dense(input_dim, 64, relu),
            Dropout(0.2),
            Dense(64, 32, relu),
            LayerNorm(32)
        )

        layers = (
            conv1 = SAGEConv(hidden_dim => output_dim),
            conv2 = SAGEConv(output_dim => output_dim)
        )

        proj_head = Chain(
            Dense(output_dim, output_dim, relu),
            Dense(output_dim, output_dim)
        )

        disc = Flux.Bilinear((output_dim, output_dim) => 1)

        return MultiViewGNN(encoder, layers, proj_head, disc, embeddings, [0.0f0], [-2.0f0], [0.0f0])
    end

    # Forward pass for the GNN, acting on `g` graph and `x` node embedding matrix
    function(mvgnn:: MultiViewGNN)(g::GNNGraph, x::AbstractMatrix{Float32})
        x′ = mvgnn.encoder(x)
        h = mvgnn.layers.conv1(g, x′)
        h = relu.(h)
        h = mvgnn.layers.conv2(g, h)
        return h
    end

    # Convenience wrappers for doing a forward pass against a weighted graph structs/vectors
    function(mvgnn::MultiViewGNN)(wg::WeightedGraph)
        view_idx = VIEW_INDEX[wg.view_type]
        embed = repeat(mvgnn.view_embeddings(view_idx), 1, nv(wg.graph))
        x =  vcat(wg.graph.ndata.topo, embed)
        mvgnn(wg.graph, x)
    end

    function(mvgnn::MultiViewGNN)(wg::WeightedGraph, x::AbstractMatrix{Float32})
        # todo this seems messy and repetitive
        view_idx = VIEW_INDEX[wg.view_type]
        embed = repeat(mvgnn.view_embeddings(view_idx), 1, nv(wg.graph))
        x = vcat(x, embed)
        mvgnn(wg.graph, x)
    end

    function(mvgnn::MultiViewGNN)(views::Vector{WeightedGraph})
        res = Vector{Matrix{Float32}}(undef, length(views))
        Threads.@threads for i in eachindex(views)
            res[i] = views[i].weight[] * mvgnn(views[i])
        end
        sum(res)
    end

end