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

end