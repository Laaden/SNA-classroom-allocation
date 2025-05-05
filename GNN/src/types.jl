module Types
    using GraphNeuralNetworks, Graphs, Flux, Adapt

    export WeightedGraph
    struct WeightedGraph
        graph::GNNGraph
        weight::Ref{Float32}
        adjacency_matrix::AbstractMatrix{<:Real}

        function WeightedGraph(adj_mat::Matrix{Int64}, weight::Float32)
            g = GNNGraph(adj_mat)

            # need to figure out which node features are the most helpful
            # degree and local_clustering seem to be important
            # relying on topological node features because using demographic or
            # academic perf can bias the model for community detection
            g.ndata.topo = hcat(
                degree_centrality(g),
                local_clustering_coefficient(g),
            )'
            return new(g, Ref(weight), Float32.(adj_mat))
        end

        function WeightedGraph(graph::GNNGraph, weight::Ref{Float32}, adjacency::AbstractMatrix{<:Real})
            return new(graph, weight, adjacency)
        end
    end

    Adapt.@adapt_structure WeightedGraph

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
        # 2-layer GraphSage GNN
        layers::NamedTuple
        # projection head taken from SimCLR,
        # seems to help contrastive loss differentiate better
        # it's only used during training, we don't apply it on the final foreward pass
        projection_head::Chain
        # Used for DGI contrastive loss
        discriminator::Flux.Bilinear
        # learned embbeddings seems to perform better than topological features
        # but need to experiment with a hybrid approach
        embedding::Flux.Embedding
    end

    Flux.@layer MultiViewGNN

    # Outer constructor for the MultiViewGNN
    function MultiViewGNN(input_dim::Int64, output_dim::Int64, n_nodes::Int64)
        layers = (
            conv1 = SAGEConv(input_dim => output_dim),
            conv2 = SAGEConv(output_dim => output_dim)
        )

        proj_head = Chain(
            Dense(output_dim, output_dim, relu),
            Dense(output_dim, output_dim)
        )

        disc = Flux.Bilinear((output_dim, output_dim) => 1)
        embed = Flux.Embedding(n_nodes, input_dim)

        return MultiViewGNN(layers, proj_head, disc, embed)
    end

    # Forward pass for the GNN, acting on `g` graph and `x` node embedding matrix
    function(mvgnn:: MultiViewGNN)(g::GNNGraph, x::AbstractMatrix)
        # todo
        # x_full = vcat(x, g.ndata.topo)
        h = mvgnn.layers.conv1(g, x)
        h = relu.(h)
        h = mvgnn.layers.conv2(g, h)
        return h
    end

    # Convenience wrappers for doing a forward pass against a weighted graph structs/vectors
    (mvgnn:: MultiViewGNN)(wg::WeightedGraph) = mvgnn(wg.graph, wg.graph.ndata.x)
    (mvgnn::MultiViewGNN)(views::Vector{WeightedGraph}) = sum(g.weight[] * mvgnn(g) for g in views)

end