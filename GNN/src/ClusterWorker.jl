module ClusterWorker

    using Leiden
    using JSON3
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

        all_edges = vcat([view["edges"] for view in parsed["views"]]...)
        unique_ids = sort(unique(vcat([e[1] for e in all_edges], [e[2] for e in all_edges])))
        node_to_index = Dict(id => i for (i, id) in enumerate(unique_ids))

        views = map(parsed["views"]) do view
            adj = zeros(Int, length(unique_ids), length(unique_ids))
            for (src, tgt) in view["edges"]
                i, j = node_to_index[src], node_to_index[tgt]
                adj[i, j] = 1
                adj[j, i] = 1
            end
            g = WeightedGraph(adj, Float32(view["weight"]), view["view_type"])
            g.graph.ndata.x = zeros(Float32, 64, size(g.graph.ndata.topo, 2))
            g
        end

        return views, unique_ids
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
    end

    export julia_main
    function julia_main()::Cint
        try
            main()
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