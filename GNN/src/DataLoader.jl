module DataLoader

    using XLSX, Graphs, GNNGraphs, GraphNeuralNetworks, Flux
    using ..Types

    # Load data from the XLSX spreadsheet
    # Won't be needed when we have a DB connection later
    function matrix_from_sheet(source::String, sheet::String)
        tbl = XLSX.readtable(source, sheet).data
        tbl[1] = [x isa Int ? x : tryparse(Int, x) for x in tbl[1]]
        tbl[2] = [x isa Int ? x : tryparse(Int, x) for x in tbl[2]]
        hcat(tbl[1], tbl[2])
    end


    # For the multi-view rep. learning to work, the adjacency
    # matrices have to be the same dimensions, and have each node
    # correspond to the same node on each graph
    # This makes sure that each adjacency matrix is the same structure
    # :)
    function create_adjacency_matrix(mats::Vararg{AbstractMatrix{<:Integer}})
        all_ids = vcat((mat[:] for mat in mats)...) |>
                unique |>
                sort
        node_to_index = Dict(n => i for (i, n) in enumerate(all_ids))
        n = length(all_ids)
        index_to_node = Dict(i => n for (n, i) in node_to_index)

        adj_mtxs = []

        for mat in mats
            adj_mtx = zeros(Int64, length(all_ids), length(all_ids))
            for row in eachrow(mat)
                i = node_to_index[row[1]]
                j = node_to_index[row[2]]
                adj_mtx[i, j] = 1
            end
            push!(adj_mtxs, adj_mtx)
        end
        return Tuple(adj_mtxs), index_to_node
    end

    export load_views_and_composite
    function load_views_and_composite(xlsx_file::String)
        (adjacency_matrices, index_to_node) = create_adjacency_matrix(
            matrix_from_sheet(xlsx_file, "net_0_Friends"),
            matrix_from_sheet(xlsx_file, "net_1_Influential"),
            matrix_from_sheet(xlsx_file, "net_2_Feedback"),
            matrix_from_sheet(xlsx_file, "net_3_MoreTime"),
            matrix_from_sheet(xlsx_file, "net_4_Advice"),
            matrix_from_sheet(xlsx_file, "net_5_Disrespect"),
            matrix_from_sheet(xlsx_file, "net_affiliation_0_SchoolActivit"),
        )

        fr_mat, inf_mat, fd_mat, mt_mat, ad_mat, ds_mat, sc_mat = adjacency_matrices

        # todo, add GPU later if needed
        graph_views = [
            WeightedGraph(fr_mat, 1.0f0, "friendship"),#0.4f0),
            WeightedGraph(inf_mat, 1.0f0, "influence"),#0.6f0),
            WeightedGraph(fd_mat, 1.0f0, "feedback"),#0.8f0),
            WeightedGraph(mt_mat, 1.0f0, "more_time"),#1.0f0),
            WeightedGraph(ad_mat, 1.0f0, "advice"),#0.9f0),
            WeightedGraph(ds_mat, -1.0f0, "disrespect"),#-1.0f0),
            WeightedGraph(sc_mat, 1.0f0, "affiliation"),#0.1f0),
        ]

        composite_graph = reduce(
            (graph, (edge, weight)) -> add_edges(graph, (edge[1], edge[2], fill(weight, length(edge[1])))),
            zip(
                [edge_index(cpu(g.graph)) for g in graph_views],
                [g.weight[] for g in graph_views]
            ),
            init=GNNGraph()
        )

        return graph_views, composite_graph, index_to_node
    end

end