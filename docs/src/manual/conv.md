# Convolution Layers

## Graph Convolutional Layer

```math
X' = \sigma(\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} X \Theta)
```

where ``\hat{A} = A + I``, ``A`` denotes the adjacency matrix, and
``\hat{D} = [\hat{d}_{ij}] = \sum_{j=0} [\hat{a}_{ij}]`` is degree matrix.

```@docs
GCNConv
```

Reference: [Semi-supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

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

Reference: [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375)

---

## Graph Neural Network Layer

```math
\textbf{x}_i' = \Theta_1 \textbf{x}_i + \sum_{j \in \mathcal{N}(i)} \Theta_2 \textbf{x}_j
```

```@docs
GraphConv
```

Reference: [Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks](https://arxiv.org/abs/1810.02244)

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

Reference: [Graph Attention Networks](https://arxiv.org/abs/1710.10903)

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

Reference: [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493)

---

## Edge Convolutional Layer

```math
\textbf{x}_i' = \sum_{j \in \mathcal{N}(i)} f_{\Theta}(\textbf{x}_i || \textbf{x}_j - \textbf{x}_i)
```

where ``f_{\Theta}`` denotes a neural network parametrized by ``\Theta``, *i.e.*, a MLP.

```@docs
EdgeConv
```

Reference: [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829)
