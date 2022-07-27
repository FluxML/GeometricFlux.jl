# Graph Convolutional Layers

## Graph Convolutional Layer

```math
X' = \sigma(\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} X \Theta)
```

where ``\hat{A} = A + I``, ``A`` denotes the adjacency matrix, and
``\hat{D} = [\hat{d}_{ij}] = \sum_{j=0} [\hat{a}_{ij}]`` is degree matrix.

```@docs
GCNConv
```

Reference: [Kipf2017](@cite)

---

## Chebyshev Spectral Graph Convolutional Layer

```math
X' = \sum^{K-1}_{k=0} Z^{(k)} \Theta^{(k)}
```

where ``Z^{(k)}`` is the ``k``-th term of Chebyshev polynomials, and can be calculated by the following recursive form:

```math
Z^{(0)} = X \\
Z^{(1)} = \hat{L} X \\
Z^{(k)} = 2 \hat{L} Z^{(k-1)} - Z^{(k-2)}
```

and ``\hat{L} = \frac{2}{\lambda_{max}} L - I``.

```@docs
ChebConv
```

Reference: [Defferrard2016](@cite)

---

## Graph Neural Network Layer

```math
\textbf{x}_i' = \sigma (\Theta_1 \textbf{x}_i + \sum_{j \in \mathcal{N}(i)} \Theta_2 \textbf{x}_j)
```

```@docs
GraphConv
```

Reference: [Morris2019](@cite)

---

## SAmple and aggreGatE (GraphSAGE) Network

```math
\hat{\textbf{x}}_j = sample(\textbf{x}_j), \forall j \in \mathcal{N}(i) \\
\textbf{m}_i = aggregate(\hat{\textbf{x}}_j) \\
\textbf{x}_i' = \sigma (\Theta_1 \textbf{x}_i + \Theta_2 \textbf{m}_i)
```

```@docs
SAGEConv
MeanAggregator
MeanPoolAggregator
MaxPoolAggregator
LSTMAggregator
```

Reference: [Hamilton2017](@cite) and [GraphSAGE website](http://snap.stanford.edu/graphsage/)

---

## Graph Attentional Layer

```math
\textbf{x}_i' = \alpha_{i,i} \Theta \textbf{x}_i + \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \Theta \textbf{x}_j
```

where the attention coefficient ``\alpha_{i,j}`` can be calculated from

```math
\alpha_{i,j} = \frac{exp(LeakyReLU(\textbf{a}^T [\Theta \textbf{x}_i || \Theta \textbf{x}_j]))}{\sum_{k \in \mathcal{N}(i) \cup \{i\}} exp(LeakyReLU(\textbf{a}^T [\Theta \textbf{x}_i || \Theta \textbf{x}_k]))}
```

```@docs
GATConv
```

Reference: [GAT2018](@cite)

---

## Graph Attentional Layer v2


```@docs
GATv2Conv
```

Reference: [Brody2022](@cite)

---

## Gated Graph Convolution Layer

```math
\textbf{h}^{(0)}_i = \textbf{x}_i || \textbf{0} \\
\textbf{h}^{(l)}_i = GRU(\textbf{h}^{(l-1)}_i, \sum_{j \in \mathcal{N}(i)} \Theta \textbf{h}^{(l-1)}_j)
```

 where ``\textbf{h}^{(l)}_i`` denotes the ``l``-th hidden variables passing through GRU. The dimension of input ``\textbf{x}_i`` needs to be less or equal to `out`.

```@docs
GatedGraphConv
```

Reference: [Li2016](@cite)

---

## Edge Convolutional Layer

```math
\textbf{x}_i' = \sum_{j \in \mathcal{N}(i)} f_{\Theta}(\textbf{x}_i || \textbf{x}_j - \textbf{x}_i)
```

where ``f_{\Theta}`` denotes a neural network parametrized by ``\Theta``, *i.e.*, a MLP.

```@docs
EdgeConv
```

Reference: [Wang2019](@cite)

---

## Graph Isomorphism Network

```math
\textbf{x}_i' = f_{\Theta}\left((1 + \varepsilon) \cdot \textbf{x}_i + \sum_{j \in \mathcal{N}(i)} \textbf{x}_j \right)
```

where ``f_{\Theta}`` denotes a neural network parametrized by ``\Theta``, *i.e.*, a MLP.

```@docs
GINConv
```

Reference: [Xu2019](@cite)

---

## Crystal Graph Convolutional Network

```math
\textbf{x}_i' = \textbf{x}_i + \sum_{j \in \mathcal{N}(i)} \sigma\left( \textbf{z}_{i,j} \textbf{W}_f + \textbf{b}_f \right) \odot \text{softplus}\left(\textbf{z}_{i,j} \textbf{W}_s + \textbf{b}_s \right)
```

where ``\textbf{z}_{i,j} = [\textbf{x}_i, \textbf{x}_j}, \textbf{e}_{i,j}]`` denotes the concatenation of node features, neighboring node features, and edge features. The operation ``\odot`` represents elementwise multiplication, and ``\sigma`` denotes the sigmoid function.

```@docs
CGConv
```

Reference: [Xie2018](@cite)
