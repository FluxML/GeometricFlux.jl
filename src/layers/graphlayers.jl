"""
    AbstractGraphLayer

An abstract type of graph neural network layer for GeometricFlux.
"""
abstract type AbstractGraphLayer end

(l::AbstractGraphLayer)(x::AbstractMatrix) = l(l.fg, x)
(l::AbstractGraphLayer)(::NullGraph, x::AbstractMatrix) = throw(ArgumentError("concrete FeaturedGraph is not provided."))
