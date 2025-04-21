module Types
    using GraphNeuralNetworks, Graphs, Flux

    export WeightedGraph
    struct WeightedGraph
        graph::GNNGraph
        weight::Array{Real}
        function WeightedGraph(adjacency_matrix::Matrix{Int64}, weight::Real)
            g = GNNGraph(adjacency_matrix) |> gpu
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
            return new(g, [weight])
        end
    end
end