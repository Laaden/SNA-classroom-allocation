using Statistics, Graphs, Flux, GraphNeuralNetworks, DataFrames, CUDA
include.(["./data_loader.jl", "./types.jl", "./loss.jl"])

# ~~ Read in data and setup data structures ~~ #

xlsx_file = "./data/Student Survey - Jan.xlsx"

friendships_mat::Matrix{Int64} = matrix_from_sheet(xlsx_file, "net_0_Friends")
disrespect_mat::Matrix{Int64} = matrix_from_sheet(xlsx_file, "net_5_Disrespect")
friend_adj_mtx, disrespect_adj_mtx = create_adjacency_matrix(
	friendships_mat,
	disrespect_mat
)

g1 = WeightedGraph(friend_adj_mtx, 1)
g2 = WeightedGraph(disrespect_adj_mtx, -0.5)

# ~~ Setup Model ~~ #

model = GNNChain(
    SAGEConv(1 => 128),
    SAGEConv(128 => 128)
)


# ~~ Train Model ~~ #
# Should use DGI or some form of contrastive loss
# Use CUDA to offload training to the GPU, more performant

epochs = 100

for epoch in 1:epochs
	# train
end


# ~~ Model Output ~~ #

output1 = model(g1.graph, g1.graph.ndata.x)
output2 = model(g2.graph, g2.graph.ndata.x)


# ~~ Aggregation ~~ #
# This could technically end up as an algo as well
output = (g1.weight * output1) + (g2.weight * output2)

# ~~ Pass this off to community detection ~~ #

# E.g. k-means
# (I don't think we're using kmeans but it's illustrative)
#
# using Clustering
# k::Integer = 20
# clusters = kmeans(output, k, maxiter=100)

# ~~ PSO ~~ #
# do some PSO stuff at some point for class size & other node features