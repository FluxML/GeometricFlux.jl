# Models

## Autoencoders

### Graph Autoencoder

```math
Z = enc(X, A) \\
\hat{A} = \sigma (ZZ^T)
```

where ``A`` denotes the adjacency matrix.

```@docs
GeometricFlux.GAE
```

Reference: [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308)

---

### Variational Graph Autoencoder

```math
H = enc(X, A) \\
Z_{\mu}, Z_{logσ} = GCN_{\mu}(H, A), GCN_{\sigma}(H, A) \\
\hat{A} = \sigma (ZZ^T)
```

where ``A`` denotes the adjacency matrix, ``X`` denotes node features.

```@docs
GeometricFlux.VGAE
```

Reference: [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308)

---

## DeepSet

```math
Z = \rho ( \sum_{x_i \in \mathcal{V}} \phi (x_i) )
```

where ``\phi`` and ``\rho`` denote two neural networks and ``x_i`` is the node feature for node ``i``.

```@docs
GeometricFlux.DeepSet
```

Reference: [Deep Sets](https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)

---

## Special Layers

### Inner-product Decoder

```math
\hat{A} = \sigma (ZZ^T)
```

where ``Z`` denotes the input matrix from encoder.

```@docs
GeometricFlux.InnerProductDecoder
```

Reference: [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308)

---

### Variational Graph Encoder

```math
H = enc(X, A) \\
Z_{\mu}, Z_{logσ} = GCN_{\mu}(H, A), GCN_{\sigma}(H, A)
```

```@docs
GeometricFlux.VariationalGraphEncoder
```

Reference: [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308)
