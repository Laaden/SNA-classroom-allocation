using Test
using GNNProject, Flux

adj = [
    0 1 1 0;
    1 0 1 0;
    1 1 0 1;
    0 0 1 0
]
wg = WeightedGraph(adj, 1.0f0, "friendship")
wg_neg = WeightedGraph(adj, -1.0f0, "disrespect")
model = MultiViewGNN(size(wg.graph.ndata.topo, 1), 16)

function has_nonzero_grads(grads)
    for g in grads
        if g !== nothing
            if isa(g, AbstractArray)
                if !all(iszero, g)
                    return true
                end
            elseif isa(g, NamedTuple) || isa(g, Tuple)
                if has_nonzero_grads(values(g))
                    return true
                end
            end
        end
    end
    return false
end


@testset "Loss Tests" begin
    @testset "Contrast" begin
        for g in [wg, wg_neg]
            loss, acc = contrastive_loss(g, g.graph.ndata.topo, model)
            # Loss should be a scalar, and not be infinite/nan/etc
            @test isfinite(loss)
            # accuracy should be bounded between 0 & 1
            @test 0.0f0 ≤ acc ≤ 1.0f0
        end
    end

    @testset "Modularity" begin
        mod_loss = soft_modularity_loss(wg, wg.graph.ndata.topo, model)
        @test isfinite(mod_loss)
        @test isa(mod_loss, Number)

        grad = Flux.gradient(
            m -> soft_modularity_loss(wg, wg.graph.ndata.topo, m),
            model
        )
        @test has_nonzero_grads(values(grad))

    end
end