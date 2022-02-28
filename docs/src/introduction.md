# Introduction to Graph Neural Networks (GNN)

Graph neural networks act as standalone network architecture other than convolutional neural networks (CNN), recurrent neural networks (RNN). As its name implies, GNN needs a graph as training data. The problem setting requires at least a graph to train on.

## What is Graph Neural Networks?

Graph convolutional layers is the building block for GNN, and it extends from classic convolutional layer. Convolutional layer performs convolution operation over regular grid geometry, e.g. pixels in images arrange regularly along vertical and horizontal directions, while graph convolutional performs convolutional operation over irregular graph topology, e.g. graph is composed of a set of *nodes* which connect to each other with *edges*. In signal processing field, images are viewed as a kind of signals. Precisely, image can be represented as a function which maps from coordinates to color for each pixel. A matrix satisfies the definition and it maps image indices to a RGB value representing each pixel. Analogically, a graph signal can be defined as a function which maps from node/edge in a graph to certain value or features. We call them node features or edge features if the features correspond to node or edge, respectively. Graph convolutional layer maps features on nodes or edges to their embeddings.

```@raw html
<figure>
    <img src="../assets/geometry.svg" width="50%" alt="geometry.svg" /><br>
    <figcaption><em>Geometry for images and graphs.</em></figcaption>
</figure>
```

## What is the Difference between Deep Learning and GNN?

Practically, GNN requires graph to be input in a certain form and features will be mapped according to the input graph, while classic deep learning architecture doesn't require a graph or a geometric object as input. In the design of GeometricFlux, the input graph can be two kinds: static graph or variable graph. A static graph is carried within a GNN layer, while a variable graph can be carried with features. The concept of a static graph defines the graph topology in the GNN layer and view it as a built-in static topology for a layer. The concept of variable graph is, totally different from static graph, to consider graph as a part of input data, which is more nature to most of people.

## Features for GNNs

Graph signals include node signals, edge signals and global (or graph) signals. According to the problem setting, signals are further classified as features or labels. Features that can be used in GNN includes node features, edge features and global (or graph) features. Global (or graph) features are features that corresponds the whole graph and represents the status of a graph.

```@raw html
<figure>
    <img src="../assets/graph signals.svg" width="70%" alt="graph signals.svg" /><br>
    <figcaption><em>Signals and graph signals.</em></figcaption>
</figure>
```
