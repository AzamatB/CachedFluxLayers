using LinearAlgebra

import Flux
import ZygoteRules

struct CachedDense{M <: AbstractVecOrMat, V <: AbstractVecOrMat, F, C <: AbstractVecOrMat}
   W :: M
   b :: V
   σ :: F
   cache :: C
end

Flux.trainable(m::CachedDense) = (m.W, m.b, m.σ)

Flux.@functor CachedDense

CachedDense(W, b, cache) = CachedDense(W, b, identity, cache)

function CachedDense(in::Integer, out::Integer, σ = identity; batchsize::Integer,
               initW = Flux.glorot_uniform, initb = Flux.zeros)
   return CachedDense(initW(out, in), initb(out), σ, Flux.zeros(out, batchsize))
end

function Base.show(io::IO, m::CachedDense)
   out, in = size(m.W)
   batchsize = size(m.cache, 2)
   print(io, "CachedDense($in, $out")
   m.σ == identity || print(io, ", ", m.σ)
   print(io, "; batchsize=$batchsize)")
end

function (m::CachedDense)(x::AbstractVecOrMat)
   W, b, σ, cache = m.W, m.b, m.σ, m.cache
   Wx = mul!(cache, W, x)
   return cache .= σ.(Wx .+ b)
end

# Try to avoid hitting generic matmul in some simple cases
# Base's matmul is so slow that it's worth the extra conversion to hit BLAS
(m::CachedDense{M})(x::AbstractVecOrMat{T}) where {T <: Union{Float32,Float64}, M <: AbstractVecOrMat{T}} =
   invoke(m, Tuple{AbstractVecOrMat}, x)

(m::CachedDense{M})(x::AbstractVecOrMat{<:AbstractFloat}) where {T <: Union{Float32,Float64}, M <: AbstractVecOrMat{T}} =
   m(T.(x))
