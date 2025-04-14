using GraphNeuralNetworks, Graphs

# Using degree centrality here but we could use anything
struct WeightedGraph
    graph::GNNGraph
    weight::Real
    function WeightedGraph(adjacency_matrix::Matrix{Int64}, weight::Real)
        if -1 <= weight <= 1
            g = GNNGraph(adjacency_matrix)
            g.ndata.x = degree_centrality(g)'
            return new(g, weight)
        else
            error("Weight must be between 1 and -1")
        end
    end
end
