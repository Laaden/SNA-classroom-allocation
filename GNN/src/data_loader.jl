module DataLoader

    using XLSX

    # Load data from the XLSX spreadsheet
    # Won't be needed when we have a DB connection later
    export matrix_from_sheet
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
    export create_adjacency_matrix
    function create_adjacency_matrix(mats::Vararg{AbstractMatrix{<:Integer}})
        all_ids = vcat((mat[:] for mat in mats)...) |>
                unique |>
                sort
        node_to_index = Dict(n => i for (i, n) in enumerate(all_ids))
        n = length(all_ids)

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
        return Tuple(adj_mtxs)
    end

end