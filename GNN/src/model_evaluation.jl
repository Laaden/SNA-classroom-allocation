
module ModelEvaluation
    using Statistics, Graphs, GNNGraphs, Clustering, Distances

    function cluster_conductance(graph::GNNGraph, labels::Vector{Int}, k::Int)
        nodes = findall(node -> node == k, labels) |>
                Set
        neighbour_list = neighbors.(Ref(graph), nodes)

        internal = mapreduce(
            neighbours -> count(node -> node in nodes, neighbours),
            +,
            neighbour_list
        )
        boundary = mapreduce(
            neighbours -> count(node -> node ∉ nodes, neighbours),
            +,
            neighbour_list
        )

        volume = internal + boundary

        return volume == 0 ? 0 : boundary / (volume)
    end

    const AcceptedMetrics = Union{[
            SqEuclidean,
            Euclidean,
            CosineDist,
            CorrDist,
            Cityblock,
            Chebyshev
        ]...
    }


    # how similar is a node to its own community, compared
    # to other communities?
    # this function takes GNN node embeddings, and
    # a vector of assignments (e.g. from kmeans or leiden)
    # Scores range from -1 to 1
    export embedding_metrics
    function embedding_metrics(
        graph::GNNGraph,
        embeddings::Matrix{Float64},
        labels::Vector{Int64},
        metric::AcceptedMetrics = SqEuclidean()
    )
        dist_mtx = pairwise(metric, embeddings, embeddings)
        metrics = Dict(
            :silhouettes => clustering_quality(labels, dist_mtx; quality_index=:silhouettes),
            :modularity => modularity(graph, labels),
            :conductance => cluster_conductance.(Ref(graph), Ref(labels), unique(labels)) |> mean,
        )
        return metrics
    end
end
