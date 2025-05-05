using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GNNProject, HTTP, BSON, Mongoc, JSON3

@info "Loading model..."
BSON.@load "output/models/model.bson" model
model = model::MultiViewGNN
BSON.@load "output/artifacts/views.bson" views
views = views::Vector{WeightedGraph}
BSON.@load "output/artifacts/clustered_students.bson" clustered_students


HTTP.serve(req -> begin
    if (req.target == "/embeddings")
        embed = model(views)
        return HTTP.Response(200, JSON3.write(embed))
    elseif (req.target == "/")
        return HTTP.Response(200, JSON3.write(clustered_students))
    end
end, "0.0.0.0", 8080)
