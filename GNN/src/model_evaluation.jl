
module ModelEvaluation
    using Flux, Statistics, Graphs, GNNGraphs, Clustering, Distances, Leiden, LinearAlgebra

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
        embeddings::Matrix{<:Real},
        labels::Vector{<:Real},
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

    export evaluate_embeddings
    function evaluate_embeddings(embeddings, graph; k=10)
        norm_embeddings = Flux.normalise(embeddings; dims = 1)
        knn = knn_graph(norm_embeddings, k)
        clusters = leiden(adjacency_matrix(knn), "ngrb")
        return embedding_metrics(graph, norm_embeddings, clusters)
    end


    export fast_evaluate_embeddings
    function fast_evaluate_embeddings(embeddings, graph; k = Int64(round(sqrt(size(embeddings, 2)))))
        norm_embeddings = Flux.normalise(embeddings; dims = 1)
        clusters = kmeans(norm_embeddings, k)
        # return embedding_metrics(graph, norm_embeddings, clusters.assignments)
        # returns dict for symmetry with the slow version
        return Dict(
            :modularity => modularity(graph, clusters.assignments)
        )
    end

    # todo, examine GPU compat
    export intra_cluster_rate
    function intra_cluster_rate(assignments::Vector{<:Real}, graph)
        pos_intra = 0
        pos_total = 0
        neg_intra = 0
        neg_total = 0
        edge_weights = get_edge_weight(graph)

            for (i, e) in enumerate(edges(graph))
                source, target = src(e), dst(e)
                weight = edge_weights[i]

                if weight > 0
                    pos_total += 1
                    if assignments[source] == assignments[target]
                        pos_intra += 1
                    end
                elseif weight < 0
                    neg_total += 1
                    if assignments[source] == assignments[target]
                        neg_intra += 1
                    end
                end
            end

            if (neg_total > 0)
                return Dict(
                    :positive_intra => pos_intra / max(pos_total, 1),
                    :negative_intra => neg_intra / max(neg_total, 1)
                )
            else
                return Dict(
                    :intra => pos_intra / max(pos_total, 1)
                )
            end
    end

    function intra_cluster_rate(assignments::Vector{<:Real}, views::Array, names::Vector{String})
        if names == Nothing
            return [intra_cluster_rate(assignments, v.graph) for v in views]
        else
             return Dict(name => intra_cluster_rate(assignments, v.graph) for (v, name) in zip(views, names))
        end
    end
end


