module Types
    using GraphNeuralNetworks, Graphs, Flux

    export WeightedGraph
    struct WeightedGraph
        graph::GNNGraph
        weight::Ref{Float32}
        adjacency_matrix::AbstractMatrix{<:Real}

        function WeightedGraph(adj_mat::Matrix{Int64}, weight::Float32)
            g = GNNGraph(adj_mat) |> gpu
            @assert length(weight) == 1 "Weight vector must have length 1"

            mtx = adjacency_matrix(g)

            # need to figure out which node features are the most helpful
            # degree and local_clustering seem to be important
            # relying on topological node features because using demographic or
            # academic perf can bias the model for community detection
            #
            # currently not being used, but need to test if it leads to better performance
            # to combine static + learned embeddings
            #
            # g.ndata.topo = vcat(
            #     # rand(Float32, g.num_nodes)',
            #     # eigenvector_centrality(g)',
            #     # degree_centrality(g)',
            #     # betweenness_centrality(g)',
            #     local_clustering_coefficient(g)',
            #     # pagerank(g)'
            # )
            return new(g, Ref(weight), mtx)
        end
    end

    export TrainLog
    struct TrainLog
        loss::Vector{Float32}
        accuracy::Vector{Float32}
        modularity::Vector{Float32}
        silhouette::Vector{Float32}
    end

    export TrainResult
    struct TrainResult{T}
        model::T
        λ::Float32
        τ::Float32
        logs::TrainLog
    end

    export MultiViewGNN
    struct MultiViewGNN
        layers::NamedTuple
        # projection head taken from SimCLR,
        # seems to help contrastive loss differentiate better
        # it's only used during training, we don't apply it on the final foreward pass
        projection_head::Chain
        discriminator::Flux.Bilinear
        # learned embbeddings seems to perform better than topological features
        # but need to experiment with a hybrid approach
        embedding::Flux.Embedding
    end

    Flux.@layer MultiViewGNN

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

    function(mvgnn:: MultiViewGNN)(g::GNNGraph, x::AbstractMatrix)
        h = mvgnn.layers.conv1(g, x)
        h = relu.(h)
        h = mvgnn.layers.conv2(g, h)
        return h
    end

    function(mvgnn::MultiViewGNN)(views::Vector{WeightedGraph})
        sum(g.weight[] * mvgnn(g.graph, g.graph.ndata.x) for g in views)
    end

end