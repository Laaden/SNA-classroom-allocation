# README

Entry is the main.jl file. Will precompile this at some point!

Modularity is currently the target metric for GNN training. We have seen modularity between 0.05 to 0.3 depending on the run of the GNN -- stabilises between 0.2 and 0.4. Discriminator accuracy is typically around 0.75 to 0.85.

## Output

This is an example output from the forward pass of the GNN.

```julia
64×201 Matrix{Float64}:
  0.80726    -1.81148     2.30621    0.33415   1.12823    0.450889  -0.179297  -0.603575   0.300963  …  -0.876584   1.44193   0.521379   3.17712    2.53001    -1.74931    1.20379    0.739938
 -2.08236    -2.47768    -3.25508   -2.59713  -2.89663   -4.0718    -3.11804   -3.01171   -3.03384      -6.43509   -4.11064   0.218806  -4.29641   -7.7216     -4.48185   -2.18979   -5.99345
  0.327323    1.48756    -0.428106  -0.52985  -0.260636  -1.63891   -0.334128   1.08676    0.6524       -1.01003   -4.17232   1.89043   -0.429061  -5.09808    -9.34965    0.244752  -1.36397
  ⋮                                                       ⋮                                          ⋱                        ⋮                                                       ⋮
  4.68941     3.42879     2.14729    6.41692   0.601989  10.6007    -3.50109    8.30305    4.15107      -4.58428   28.3081    0.246402   0.72482   22.9906     26.2204     2.64578    7.76408
  0.0555717  -1.53523     1.21299    1.31309   0.170547  -2.75328    1.70838    0.948898  -0.75559      -3.24537    1.05892   1.58976    0.452771  -0.0761089  -1.79515    0.186128  -2.45045
 -1.14908    -0.0529032  -1.77724    1.63399  -1.92957   -0.822541  -2.81207    0.577683  -1.58878      -0.884831   0.500229  1.08064    1.51604    2.25843     0.791221  -2.6743     1.50099
```

Examining the clusters (produced via a basic, unrefined KMeans), we get evidence of repulsion & attraction working effectively. Across views the intra-cluster rate is as follows:

```julia
Dict{String, Dict{Symbol, Float64}} with 7 entries:
  "influence"   => Dict(:intra=>0.217391)
  "friendship"  => Dict(:intra=>0.246364)
  "more_time"   => Dict(:intra=>0.3)
  "advice"      => Dict(:intra=>0.310627)
  "disrespect"  => Dict(:intra=>0.0131579)
  "feedback"    => Dict(:intra=>0.356808)
  "affiliation" => Dict(:intra=>0.0347222)
```

If we compare this to the composite graph, we can see the repulsive effects more clearly:

```julia
Dict{Symbol, Float64} with 2 entries:
  :negative_intra => 0.0131579
  :positive_intra => 0.246419
```
