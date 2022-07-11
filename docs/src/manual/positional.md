# Positional Encoding

## Positional Encoding Methods

```@docs
AbstractPositionalEncoding
RandomWalkPE
LaplacianPE
positional_encode
```

## Positional Encoding Layers

### ``E(n)``-equivariant Positional Encoding Layer

It employs message-passing scheme and can be defined by following functions:

- message function: ``y_{ij}^l = (x_i^l - x_j^l)\phi_x(m_{ij})``
- aggregate: ``y_i^l = \frac{1}{M} \sum_{j \in \mathcal{N}(i)} y_{ij}^l``
- update function: ``x_i^{l+1} = x_i^l + y_i^l``

where ``x_i^l`` and ``x_j^l`` denote the positional feature for node ``i`` and ``j``, respectively, in ``l``-th layer, ``\phi_x`` is the neural network for positional encoding and ``m_{ij}`` is the edge feature for edge ``(i,j)``. ``y_{ij}^l`` and ``y_i^l`` represent the encoded positional feature and aggregated positional feature, respectively, and ``M`` denotes number of neighbors of node ``i``.

```@docs
EEquivGraphPE
```

Reference: [Satorras2021](@cite)

---

### Learnable Structural Positional Encoding layer

(WIP)

```@docs
LSPE
```

Reference: [Dwivedi2021](@cite)

---

