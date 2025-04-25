# ENV["CUDA_VISIBLE_DEVICES"] = "-1"
include("src/data_loader.jl")
include("src/types.jl")
include("src/loss.jl")
include("src/model_evaluation.jl")
include("src/training.jl")

using Statistics, Graphs, Flux, GraphNeuralNetworks, Random, Zygote, DataFrames, CUDA, Leiden, Distances, LinearAlgebra
using .DataLoader, .Types, .Loss, .ModelEvaluation, .ModelTraining

# ~~ Read in data and setup data structures ~~ #

xlsx_file = "./data/Student Survey - Jan.xlsx"

fr_mat, inf_mat, fd_mat, mt_mat, ad_mat, ds_mat, sc_mat = create_adjacency_matrix(
	matrix_from_sheet(xlsx_file, "net_0_Friends"),
    matrix_from_sheet(xlsx_file, "net_1_Influential"),
    matrix_from_sheet(xlsx_file, "net_2_Feedback"),
    matrix_from_sheet(xlsx_file, "net_3_MoreTime"),
    matrix_from_sheet(xlsx_file, "net_4_Advice"),
    matrix_from_sheet(xlsx_file, "net_5_Disrespect"),
    matrix_from_sheet(xlsx_file, "net_affiliation_0_SchoolActivit"),
)

graph_views = [
    WeightedGraph(fr_mat, 0.4),
    WeightedGraph(inf_mat, 0.6),
    WeightedGraph(fd_mat, 0.8),
    WeightedGraph(mt_mat, 1),
    WeightedGraph(ad_mat, 0.9),
    WeightedGraph(ds_mat, -0.5),
    WeightedGraph(sc_mat, 0.1),
]

composite_graph = reduce(
    (graph, (edge, weight)) -> add_edges(graph, (edge[1], edge[2], fill(weight, length(edge[1])))),
    zip(
        [edge_index(cpu(g.graph)) for g in graph_views],
        [g.weight[1] for g in graph_views]
    ),
    init=GNNGraph()
)

# ~~ Setup Model ~~ #

n_nodes = size(composite_graph, 1)
embedding_dim = 64
input_dim = embedding_dim
output_dim = 64 # typically used for comm. detection, can try 64 as well

model = GNNChain(
	SAGEConv(input_dim => output_dim),
	x -> relu.(x),
	SAGEConv(output_dim => output_dim)
) |> gpu

# projection head taken from SimCLR, seems to help contrastive loss differentiate better
# (think of it like a mini PCA)
# it's only used during training, we don't apply it on the final foreward pass
proj_head = Chain(
    Dense(output_dim, output_dim, relu),
    Dense(output_dim, output_dim)
) |> gpu

opt = Flux.Adam(1e-3) |> gpu
discriminator = Flux.Bilinear((output_dim, output_dim) => 1) |> gpu
# learned embbeddings seems to perform better than topological features
# but need to experiment with a hybrid approach
node_embedding = Flux.Embedding(n_nodes, embedding_dim) |> gpu
ps = Flux.params(model, discriminator, proj_head, node_embedding)


# ~~ Train Model ~~ #

results = hyperparameter_search(
    model,
    proj_head,
    discriminator,
    node_embedding,
    graph_views,
    composite_graph,
    taus    = [0.3, 0.4, 1],
    lambdas = [3.0, 6.0, 10],
    epochs = 500,
    n_repeats = 3
)

# given community detection is the goal,
# modularity is our best metric for optimising the GNN
best_parameters = argmax(r -> maximum(r[:modularity]), results)

trained_model = train_model(
    model,
    proj_head,
    opt,
    discriminator,
    node_embedding,
    ps,
    graph_views,
    composite_graph;
    λ=best_parameters[:λ],
    τ=best_parameters[:τ],
    verbose = true,
    epochs = 500
)

# ~~ Model Output & Aggregation ~~ #
# This could technically end up as an algo as well

output = cpu(sum((g.weight[1]) * trained_model[:model](g.graph, g.graph.ndata.x) for g in graph_views))


# # ~~ Pass this off to community detection ~~ #

# # E.g. k-means
# # (I don't think we're using kmeans but it's illustrative)
# #
# using Clustering
# k = Int64(round(sqrt(size(output, 2))))
# clusters = kmeans(normalize(output), k, maxiter=100)
