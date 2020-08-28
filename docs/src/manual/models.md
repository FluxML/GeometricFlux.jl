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
H = enc(X, A) \\
Z_{\mu}, Z_{logσ} = GCN_{\mu}(H, A), GCN_{\sigma}(H, A) \\
\hat{A} = \sigma (ZZ^T)
```

where ``A`` denotes the adjacency matrix, ``X`` denotes node features.

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
H = enc(X, A) \\
Z_{\mu}, Z_{logσ} = GCN_{\mu}(H, A), GCN_{\sigma}(H, A)
```

```@docs
VariationalEncoder
```

Reference: [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308)
