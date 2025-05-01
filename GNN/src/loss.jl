module Loss
	using ..Types
	using GNNGraphs, Flux, Graphs, Random, Statistics, Zygote, Clustering, LinearAlgebra

	Zygote.@nograd shuffle
	export contrastive_loss
	function contrastive_loss(g::WeightedGraph, x::AbstractMatrix{Float32}, model::MultiViewGNN; τ::Float32=1f0)

		# doing some DBI contrastive loss here
		h = model(g.graph, x) |>
			model.projection_head |>
			x -> Flux.normalise(x, dims = 1)

		x_neg = x[:, shuffle(1:end)]
		h_neg = model(g.graph, x_neg) |>
			model.projection_head |>
			x -> Flux.normalise(x, dims = 1)

		global_summary = mean(h, dims=2)
		summary_mat = repeat(global_summary, 1, size(h, 2)) |>
			x -> Flux.normalise(x; dims = 1)

		pos_scores = model.discriminator(h, summary_mat) / τ
		neg_scores = model.discriminator(h_neg, summary_mat) / τ

		# we treat the graph and negative graph as a binary classification
		# problem, so we create our own labels.
		# theoretically negative graphs are repulsive, so we invert the labels
		labels =
			sign(g.weight[]) == 1 ?
			gpu(vcat(ones(Float32, size(pos_scores)), zeros(Float32, size(neg_scores)))) :
			gpu(vcat(zeros(Float32, size(pos_scores)), ones(Float32, size(neg_scores))))

		scores = vcat(pos_scores, neg_scores)

		probs = sigmoid.(scores)
		preds = probs .> 0.5
  		acc = mean(preds .== Bool.(labels))

    	return Flux.logitbinarycrossentropy(scores, labels), acc

	end

	# This is adapted from a paper on modularity loss
	# "UNSUPERVISED COMMUNITY DETECTION WITH MODULARITY-BASED ATTENTION MODEL"
	#  by Ivan Lobov, Sergey Ivanov
	# https://github.com/Ivanopolo/modnet
	# It *is* differentiable
    # The algorithm is modified slightly because we are using polarity to
	# decrease modularity in the case of repulsive
	export soft_modularity_loss
	function soft_modularity_loss(g::WeightedGraph, x::AbstractMatrix{Float32}, model::MultiViewGNN)
        A = sign(g.weight[]) * Float32.(g.adjacency_matrix)
		h = Flux.softmax(model(g.graph, x); dims = 1)

		indegs = Float32.(degree(g.graph, dir=:in))
		outdegs = Float32.(degree(g.graph, dir=:out))
		m = ne(g.graph)

    	expected = (outdegs * indegs') / m
		B = A - expected

		soft_mod = sum(diag(h * B * h'))
		# todo, add a temperature value for further tuning
		return -soft_mod / m
	end

	# Adapted from modnet as well
	# This regularisation loss ensures that
	# we don't end up with degenerate clusters
	export cluster_balance_loss
	function cluster_balance_loss(g::WeightedGraph, x::AbstractMatrix{Float32}, model::MultiViewGNN)
    	h = Flux.softmax(model(g.graph, x); dims=1)
		n = nv(g.graph)
		k = round(Int, sqrt(n)) # this is a heuritic, need to assess
		ratio = 1.0f0 / k
		cluster_sums = sum(h, dims = 2) / n
		return sum((cluster_sums .- ratio).^2)
	end

	export calculate_total_loss
	function calculate_total_loss(model::MultiViewGNN, views::Vector{WeightedGraph}, x::AbstractMatrix{Float32}, τ::Float32, λ::Float32, γ::Float32)
		total_contrastive_loss = 0f0
		total_modularity_loss = 0f0
		total_balance_loss = 0f0
		total_accuracy = 0f0

		for g in views
			g.graph.ndata.x = x
			contrast_loss, disc_acc = contrastive_loss(g, x, model; τ=τ)
			if (λ != 0)
            	total_modularity_loss += soft_modularity_loss(g, x, model)
			end
			total_balance_loss += cluster_balance_loss(g, x, model)
			total_contrastive_loss += contrast_loss
			total_accuracy += disc_acc
		end

    	total_loss = total_contrastive_loss + λ * total_modularity_loss + γ * total_contrastive_loss

		return (total_loss, Dict(
			:contrast_loss => total_contrastive_loss,
			:mod_loss => total_modularity_loss,
			:balance_loss => total_balance_loss,
			:acc => total_accuracy
		))
	end

end
