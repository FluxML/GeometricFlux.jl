# Group Convolutional Layers

## ``E(n)``-equivariant Convolutional Layer

It employs message-passing scheme and can be defined by following functions:

- message function (Eq. 3 from paper): ``m_{ij} = \phi_e(h_i^l, h_j^l, ||x_i^l - x_j^l||^2, a_{ij})``
- aggregate (Eq. 5 from paper): ``m_i = \sum_j m_{ij}``
- update function (Eq. 6 from paper): ``h_i^{l+1} = \phi_h(h_i^l, m_i)``

where ``h_i^l`` and ``h_j^l`` denotes the node feature for node ``i`` and ``j``, respectively, in ``l``-th layer, as well as ``x_i^l`` and ``x_j^l`` denote the positional feature for node ``i`` and ``j``, respectively, in ``l``-th layer. ``a_{ij}`` is the edge feature for edge ``(i,j)``. ``\phi_e`` and ``\phi_h`` are neural network for edges and nodes.

```@docs
EEquivGraphConv
```

Reference: [Satorras2021](@cite)

---
