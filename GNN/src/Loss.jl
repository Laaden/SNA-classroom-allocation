module Loss
	using ..Types
	using GNNGraphs, Flux, Graphs, Random, Statistics, Zygote, Clustering, LinearAlgebra

	Zygote.@nograd shuffle
	export contrastive_loss
	function contrastive_loss(g::WeightedGraph, model::MultiViewGNN; τ::Float32=1f0)

		# doing some DBI contrastive loss here
		h = model(g) |>
			model.projection_head |>
			x -> Flux.normalise(x, dims = 1)

		x_neg = g.graph.ndata.topo[:, shuffle(1:end)]
		h_neg = model(g, x_neg) |>
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
		# todo, this needs to be wrapped in GPU
		# if i add that back in
		labels =
			sign(g.weight[]) == 1 ?
			vcat(ones(Float32, size(pos_scores)), zeros(Float32, size(neg_scores))) :
			vcat(zeros(Float32, size(pos_scores)), ones(Float32, size(neg_scores)))

		scores = vcat(pos_scores, neg_scores)

		probs = sigmoid.(scores)
		preds = probs .> 0.5
  		acc = mean(preds .== Bool.(labels))

    	return Flux.logitbinarycrossentropy(scores, labels), acc
	end

	# This is adapted from a paper on modularity loss
	# "UNSUPERVISED COMMUNITY DETECTION WITH MODULARITY-BASED ATTENTION MODEL"
	#  by Ivan Lobov, Sergey Ivanov
	# https://github.  com/Ivanopolo/modnet
	# It *is* differentiable
    # The algorithm is modified slightly because we are using polarity to
	# decrease modularity in the case of repulsive
	export soft_modularity_loss
	function soft_modularity_loss(g::WeightedGraph, model::MultiViewGNN)
    	A = sign(g.weight[]) * g.adjacency_matrix
    	# todo, i removed temperature
		h = model(g) |>
			x -> Flux.softmax(x; dims = 1)
		indegs = Float32.(degree(g.graph, dir=:in))
		outdegs = Float32.(degree(g.graph, dir=:out))
		m = ne(g.graph)

    	expected = (outdegs * indegs') / m
		B = A - expected
		soft_mod = sum((h * B) .* h)
		return -soft_mod / m
	end

	# Adapted from modnet as well
	# This regularisation loss ensures that
	# we don't end up with degenerate clusters
	export cluster_balance_loss
	function cluster_balance_loss(g::WeightedGraph, model::MultiViewGNN)
    	h = Flux.softmax(model(g); dims=1)
		n = nv(g.graph)
		k = infer_k(n) # this is a heuritic
		ratio = 1.0f0 / k
		cluster_sums = sum(h, dims = 2) / n
		return sum((cluster_sums .- ratio).^2)
	end

	function infer_k(n)
    	return round(Int, sqrt(n))
	end
	Zygote.@nograd infer_k


	export calculate_total_loss
	function calculate_total_loss(model::MultiViewGNN, views::Vector{WeightedGraph}, τ::Float32, λ::Float32, γ::Float32)
		total_contrastive_loss = 0f0
		total_modularity_loss = 0f0
		total_balance_loss = 0f0
		total_accuracy = 0f0

		for g in views
			# subgraph = sample_weighted_subgraph(g)
			subgraph = g
			contrast_loss, disc_acc = contrastive_loss(subgraph, model; τ=τ)
			if (λ != 0)
            	total_modularity_loss += soft_modularity_loss(subgraph, model)
			end
			total_balance_loss += cluster_balance_loss(subgraph, model)
			total_contrastive_loss += contrast_loss
			total_accuracy += disc_acc
		end

    	total_loss = total_contrastive_loss +
			λ * total_modularity_loss +
			γ * total_balance_loss

		return (total_loss, Dict(
			:contrast_loss => total_contrastive_loss,
			:mod_loss => total_modularity_loss,
			:balance_loss => total_balance_loss,
			:acc => total_accuracy
		))
	end

	export multitask_loss
	function multitask_loss(model::MultiViewGNN, views::Vector{WeightedGraph})
		Lc, Lm, Lb, Acc = compute_task_losses(model, views, 1.0f0)
		loss_c = 0.5f0 * exp(-2f0*model.logσ_c[1]) * Lc + model.logσ_c[1]
		loss_m = 0.5f0 * exp(-2f0*model.logσ_m[1]) * Lm + model.logσ_m[1]
		loss_b = 0.5f0 * exp(-2f0*model.logσ_b[1]) * Lb + model.logσ_b[1]
		return loss_c + loss_m + loss_b, Acc
	end

	export compute_task_losses
	function compute_task_losses(model::MultiViewGNN, views::Vector{WeightedGraph}, τ::Float32)
		Lc = 0f0
		Lm = 0f0
		Lb = 0f0
		Acc = 0f0
		for g in views
        	lc, acc = contrastive_loss(g, model, τ=τ)
			lm      = soft_modularity_loss(g, model)
			lb      = cluster_balance_loss(g, model)

			Lc  += lc
			Lm  += lm
			Lb  += lb
			Acc += acc
		end
		return Lc, Lm, Lb, Acc
	end

end
