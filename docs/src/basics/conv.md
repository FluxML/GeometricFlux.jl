# Graph convolutions

Graph convolution can be classified into spectral-based graph convolution and spatial-based graph convolution. Spectral-based graph convolution, such as `GCNConv` and `ChebConv`, performs operation on features of *whole* graph at one time. Spatial-based graph convolution, such as `GraphConv` and `GATConv`, performs operation on features of *local* graph instead. Message-passing scheme is an abstraction for spatial-based graph convolutional layers. Any spatial-based graph convolutional layer can be implemented under the framework of message-passing scheme.
