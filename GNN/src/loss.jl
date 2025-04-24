module Loss
	using Flux, Graphs, Random, Statistics, Zygote, Leiden, Clustering, LinearAlgebra

	Zygote.@nograd shuffle
	export contrastive_loss
	function contrastive_loss(x, model, discriminator, proj_head, τ, g)

		# doing some DBI style contrastive loss here
		g.graph.ndata.x = x
		h = model(g.graph, x) |>
			proj_head |>
			x -> Flux.normalise(x, dims = 1)

		x_neg = x[:, shuffle(1:end)]
		h_neg = model(g.graph, x_neg) |>
			proj_head |>
			x -> Flux.normalise(x, dims = 1)

		global_summary = mean(h, dims=2)
		summary_mat = repeat(global_summary, 1, size(h, 2)) |>
			x -> Flux.normalise(x; dims = 1)

		pos_scores = discriminator(h, summary_mat) / τ
		neg_scores = discriminator(h_neg, summary_mat) / τ

		# we treat the graph and negative graph as a binary classification
		# problem, so we create our own labels.
		# theoretically negative graphs are repulsive, so we invert the labels
		labels =
			sign(g.weight[1]) == 1 ?
			gpu(vcat(ones(Float32, size(pos_scores)), zeros(Float32, size(neg_scores)))) :
			gpu(vcat(zeros(Float32, size(pos_scores)), ones(Float32, size(neg_scores))))

		scores = vcat(pos_scores, neg_scores)

		probs = sigmoid.(scores)
		preds = probs .> 0.5
  		acc = mean(preds .== labels)

    	return Flux.logitbinarycrossentropy(scores, labels), acc

	end

	# This is adapted from a paper on modularity loss
	# "UNSUPERVISED COMMUNITY DETECTION WITH MODULARITY-BASED ATTENTION MODEL"
	#  by Ivan Lobov, Sergey Ivanov
	# https://github.com/Ivanopolo/modnet
	# It *is* differentiable
    #
	#
	export soft_modularity_loss
	function soft_modularity_loss(model, g, x)
		g.graph.ndata.x = x
        A = Float32.(adjacency_matrix(g.graph))
		h = Flux.softmax(model(g.graph, x); dims = 1)

		indegs = Float32.(degree(g.graph, dir=:in))
		outdegs = Float32.(degree(g.graph, dir=:out))
		m = ne(g.graph)

    	expected = (outdegs * indegs') / m
		B = A - expected

		soft_mod = sum(diag(h * B * h'))
		return -soft_mod / m
	end


	# not differentiable
	export hard_modularity_loss
	function hard_modularity_loss(x, model, g)
		h = cpu(model(g.graph, x))
		k = Int64(round(sqrt(size(h, 2))))
		clusters = kmeans(h, k)
		return -modularity(cpu(g.graph), clusters.assignments)
	end
	Zygote.@nograd hard_modularity_loss

end