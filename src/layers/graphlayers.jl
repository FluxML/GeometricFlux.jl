abstract type AbstractGraphLayer end

(l::AbstractGraphLayer)(x::AbstractMatrix) = l(l.fg, x)
