# README

[![Build GNNWorker](https://github.com/DannyTheSober/SNA-classroom-allocation/actions/workflows/release.yml/badge.svg)](https://github.com/DannyTheSober/SNA-classroom-allocation/actions/workflows/release.yml)

Modularity is currently the target metric for GNN training.
We have seen modularity between 0.05 to 0.3 depending on the run of the GNN -- stabilises between 0.2 and 0.4. Discriminator accuracy is typically around 0.75 to 0.85.

Model needs to be trained against the full dataset. At present, this is just testing of the model architecture.

## Output

Examining the clusters (produced via a basic, unrefined KMeans), we get evidence of repulsion & attraction working effectively. If we compare this to the composite graph, we can see the repulsive effects more clearly. See output/logs for more metrics.
