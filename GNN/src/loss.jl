module Loss
	using Flux, Graphs, Random, Statistics, Zygote

	Zygote.@nograd shuffle
	export contrastive_loss
	function contrastive_loss(node_embedding, model, discriminator, g)
		node_ids = 1:nv(g.graph)

		x = node_embedding(node_ids)
		# doing some DBI style contrastive loss here
		g.graph.ndata.x = x
		h = model(g.graph, x)
		x_neg = x[:, shuffle(1:end)]
		h_neg = model(g.graph, x_neg)

		global_summary = mean(h, dims=2)
		summary_mat = repeat(global_summary, 1, size(h, 2))

		pos_scores = discriminator(h, summary_mat)
		neg_scores = discriminator(h_neg, summary_mat)

		# we treat the graph and negative graph as a binary classification
		# problem
		# so, we create our own labels :)
		# theoretically for negative graphs, we are making people without ties closer
		# than those with ties, so we invert the labels
		labels =
			sign(g.weight) == 1 ?
			gpu(vcat(ones(Float32, size(pos_scores)), zeros(Float32, size(neg_scores)))) :
			gpu(vcat(zeros(Float32, size(pos_scores)), ones(Float32, size(neg_scores))))

		scores = vcat(pos_scores, neg_scores)
    	return Flux.logitbinarycrossentropy(scores, labels)

	end

end