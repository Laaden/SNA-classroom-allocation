module ClusterWorker

    using Leiden
    using JSON3
    using Base.Threads
    using SparseArrays
    import GNNGraphs: knn_graph, adjacency_matrix
    import BSON: @load
    import ArgParse: ArgParseSettings, parse_args, @add_arg_table!
    import Flux: normalise
    import ..Types: WeightedGraph

    # todo, you should be able to "pin" students
    # or should that only be outside the GNN? Maybe because the GA will try to optimise
    # the clusters, so maybe even after that.
    function load_views_from_stdin()
        raw_json = read(stdin, String)

        parsed = JSON3.read(raw_json)

        all_edges = Iterators.flatten(view["edges"] for view in parsed["views"]) |> collect

        unique_ids = sort(unique(vcat([e[1] for e in all_edges], [e[2] for e in all_edges])))
        node_to_index = Dict(id => i for (i, id) in enumerate(unique_ids))

        parsed_views = parsed["views"]
        n = length(parsed_views)
        n_nodes = length(unique_ids)
        views = Vector{WeightedGraph}(undef, n)

        Threads.@threads for i in 1:n
            view = parsed_views[i]
            adj = build_sparse_adj(view["edges"], node_to_index, n_nodes)
            g = WeightedGraph(adj, Float32(view["weight"]), view["view_type"])
            views[i] = g
        end

        return views, unique_ids
    end

    function build_sparse_adj(edges, node_to_index::Dict{Int, Int}, n::Int)::SparseMatrixCSC{Int, Int}
        m = 2 * length(edges)
        I = Vector{Int}(undef, m)
        J = Vector{Int}(undef, m)
        @inbounds for (k, (src, tgt)) in enumerate(edges)
            i = node_to_index[src]
            j = node_to_index[tgt]
            I[2k - 1] = i
            J[2k - 1] = j
            I[2k] = j
            J[2k] = i
        end
        return sparse(I, J, ones(Int, m), n, n)
    end

    export main
    function main()
        @info "Starting worker..."
        # fix not portable
        MODEL_PATH = joinpath(@__DIR__, "..", "output", "models", "model.bson")

        s = ArgParseSettings(
            prog = "GNNProject.jl",
            description = "Takes in adjacency data to create Leiden clusters from a trained GNN"
        )

        @add_arg_table! s begin
            "--model-path"; arg_type = String; default = MODEL_PATH
            "--k"; arg_type = Int; default = 10
            "--stdin"; action = :store_true; help = "Read input JSON from stdin"
        end

        args = parse_args(s)

        if !args["stdin"]
            error("No stdin flag given.")
        end

        @load args["model-path"] model

        # Profile.@profile begin
            views, node_ids = load_views_from_stdin()

            embeddings = model(views)
            norm_embeddings = normalise(embeddings; dims=1)
            n_nodes = size(norm_embeddings, 2)
            k = max(1, min(args["k"], n_nodes - 1))
            knn = knn_graph(norm_embeddings, k)
            clusters = leiden(adjacency_matrix(knn), "ngrb")

            JSON3.write(stdout, Dict(
                "assignments" => [
                    Dict("id" => node_id, "cluster" => c)
                    for (node_id, c) in zip(node_ids, clusters)
                ]
            ))
        # end
    end


    export julia_main
    function julia_main()::Cint
        try
            # @info "Profiling..."
            # Profile.init(n = 1_000_000, delay = 0.0005)
            # Profile.clear()
            main()
            # open("profile_output.txt", "w") do io
            #     Profile.print(io)
            # end
            return 0
        catch err
            @error "Error: $err"
            return 1
        end
    end

    if abspath(PROGRAM_FILE) == @__FILE__
        julia_main()
    end
end