module Types
    using GraphNeuralNetworks, Graphs, Flux

    export WeightedGraph
    struct WeightedGraph
        graph::GNNGraph
        weight::AbstractVector{<:Real}
        adjacency_matrix::AbstractMatrix{<:Real}

        function WeightedGraph(adj_mat::Matrix{Int64}, weight::Real)
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
            return new(g, [weight], mtx)
        end
    end
end