ENV["CUDA_VISIBLE_DEVICES"] = "1"

include.(["./src/data_loader.jl", "./src/types.jl", "./src/loss.jl", "./src/model_evaluation.jl"])

using Statistics, Graphs, Flux, GraphNeuralNetworks, Random, Zygote, DataFrames, CUDA
using .DataLoader, .Types, .Loss, .ModelEvaluation

# Use CUDA to offload training to the GPU, more performant
# uncomment to use cpu instead


# ~~ Read in data and setup data structures ~~ #

xlsx_file = "./data/Student Survey - Jan.xlsx"

friend_adj_mtx, moretime_adj_mtx, disrespect_adj_mtx, influential_adj_mtx = create_adjacency_matrix(
    matrix_from_sheet(xlsx_file, "net_0_Friends"),
    matrix_from_sheet(xlsx_file, "net_3_MoreTime"),
    matrix_from_sheet(xlsx_file, "net_1_Influential"),
    matrix_from_sheet(xlsx_file, "net_5_Disrespect"),
)

graph_views = [
    WeightedGraph(friend_adj_mtx, 0.5),
    WeightedGraph(moretime_adj_mtx, 1),
    WeightedGraph(disrespect_adj_mtx, -1),
    WeightedGraph(influential_adj_mtx, 0.8)
]

# ~~ Setup Model ~~ #

# input_dim = size(graph_views[1].graph.ndata.x, 1) # grab the feature size
n_nodes = size(friend_adj_mtx, 1)
embedding_dim = 64
input_dim = embedding_dim
output_dim = 128 # typically used for comm. detection, can try 64 as well

model = GNNChain(
	SAGEConv(input_dim => output_dim),
	x -> relu.(x),
	SAGEConv(output_dim => output_dim)
) |> gpu

opt = gpu(Flux.Adam(1e-3))
discriminator = gpu(Flux.Bilinear((output_dim, output_dim) => input_dim))
# learned embbeddings seems to perform better than topological features
# but need to experiment with a hybrid approach
node_embedding = gpu(Flux.Embedding(n_nodes, embedding_dim))
ps = gpu(Flux.params(model, discriminator, node_embedding))


# ~~ Train Model ~~ #
# Should use DGI or some form of contrastive loss

epochs = 300

for epoch in 1:epochs
	loss_total = 0.0
	grads = Flux.gradient(ps) do
		loss_epoch = 0.0f0
		for g in graph_views
            loss_epoch += contrastive_loss(
				node_embedding,
				model,
				discriminator,
				g
			)
		end
		loss_total = loss_epoch
		return loss_epoch
	end
	Flux.Optimise.update!(opt, ps, grads)

	if epoch % 10 == 0
		println("Epoch $epoch | Loss: $(loss_total)")
	end
end

# ~~ Model Output & Aggregation ~~ #
# This could technically end up as an algo as well
output = cpu(sum(abs(g.weight[1]) * model(g.graph, g.graph.ndata.x) for g in graph_views))


# # ~~ Pass this off to community detection ~~ #

# # E.g. k-means
# # (I don't think we're using kmeans but it's illustrative)
# #
# using Clustering
# k = Int64(round(sqrt(size(output, 2))))
# clusters = kmeans(normalize(output), k, maxiter=100)


# # ~~ PSO ~~ #
# # do some PSO stuff at some point for class size & other node features


# using Leiden

# composite_graph = reduce(
# 	(graph, edge) -> add_edges(graph, edge), [edge_index(g.graph) for g in graph_views],
# 	init=GNNGraph()
# )
# output_knn = knn_graph(
#     normalize(output),
#     # have to figure out a reasonable number
#     Int64(round(log(size(output, 2))))
# )

# leid = leiden(adjacency_matrix(output_knn), "ngrb")

# ModelEvaluation.evaluate_output(composite_graph, normalize(output), clusters.assignments)
# ModelEvaluation.evaluate_output(composite_graph, normalize(output), leid, Euclidean())
