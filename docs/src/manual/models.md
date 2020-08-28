# Models

## Autoencoders

### Graph Autoencoder

```math
Z = enc(X, A) \\
\hat{A} = \sigma (ZZ^T)
```

where ``A`` denotes the adjacency matrix.

```@docs
GAE
```

Reference: [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308)

---

### Variational Graph Autoencoder

```math
X' = \sigma(\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} X \Theta)
```

where ``\hat{A} = A + I``, ``A`` denotes the adjacency matrix, and
``\hat{D} = [\hat{d}_{ij}] = \sum_{j=0} [\hat{a}_{ij}]`` is degree matrix.

```@docs
VGAE
```

Reference: [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308)

---

## Special Layers

### Inner-product Decoder

```math
\hat{A} = \sigma (ZZ^T)
```

where ``Z`` denotes the input matrix from encoder.

```@docs
InnerProductDecoder
```

Reference: [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308)

---

### Variational Encoder

```math
H = f(X)
μ, logσ = μ(H), Σ(H)
```

```@docs
VariationalEncoder
```

Reference: [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308)
