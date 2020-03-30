using LinearAlgebra
using Flux

import Flux
import ZygoteRules

struct CachedDense{M <: AbstractVecOrMat, V <: AbstractVecOrMat, F}
   W :: M
   b :: V
   σ :: F
   cache :: NamedTuple{(:Wx, :y, :x̄, :W̄, :b̄), Tuple{M,M,M,M,V}}
end

Flux.trainable(m::CachedDense) = (m.W, m.b)

Flux.@functor CachedDense

CachedDense(W, b, cache) = CachedDense(W, b, identity, cache)

function CachedDense(in::Integer, out::Integer, σ = identity; batchsize::Integer,
               initW = Flux.glorot_uniform, initb = Flux.zeros)
   cache = (Wx = Flux.zeros(out, batchsize),
            y  = Flux.zeros(out, batchsize),
            x̄  = Flux.zeros(in, batchsize),
            W̄  = Flux.zeros(out, in),
            b̄  = Flux.zeros(out))
   return CachedDense(initW(out, in), initb(out), σ, cache)
end

function Base.show(io::IO, m::CachedDense)
   out, in = size(m.W)
   batchsize = size(m.cache.y, 2)
   print(io, "CachedDense($in, $out")
   m.σ == identity || print(io, ", ", m.σ)
   print(io, "; batchsize=$batchsize)")
end

(m::CachedDense)(x::AbstractVecOrMat) = cached_dense(x, m.W, m.b, m)
function cached_dense(x::AbstractVecOrMat, W::AbstractVecOrMat, b::AbstractVecOrMat, m::CachedDense)
   σ, c = m.σ, m.cache
   Wx = y = @view c.y[:,axes(x,2)]
   mul!(Wx, W, x)
   @. y = σ(Wx + b)
   return y
end

# Try to avoid hitting generic matmul in some simple cases
# Base's matmul is so slow that it's worth the extra conversion to hit BLAS
(m::CachedDense{M})(x::AbstractVecOrMat{T}) where {T <: Union{Float32,Float64}, M <: AbstractVecOrMat{T}} =
   invoke(m, Tuple{AbstractVecOrMat}, x)

(m::CachedDense{M})(x::AbstractVecOrMat{<:AbstractFloat}) where {T <: Union{Float32,Float64}, M <: AbstractVecOrMat{T}} =
   m(T.(x))

ZygoteRules.@adjoint function cached_dense(x::AbstractVecOrMat, W::AbstractVecOrMat, b::AbstractVecOrMat, m::CachedDense)
   σ, c = m.σ, m.cache
   W̄, b̄, batch = c.W̄, c.b̄, axes(x,2)
   @views Wx, y, x̄ = c.Wx[:,batch], c.y[:,batch], c.x̄[:,batch]
   W̄x̄⁺b̄ = Wx⁺b = Wx
   mul!(Wx, W, x)
   @. Wx⁺b = Wx + b
   @. y = σ(Wx⁺b)
   σ′ = σ'
   function cached_dense_adjoint(ȳ)
      @. W̄x̄⁺b̄ = ȳ * σ′(Wx⁺b)
      mul!(x̄, W', W̄x̄⁺b̄)
      mul!(W̄, W̄x̄⁺b̄, x')
      sum!(b̄, W̄x̄⁺b̄)
      return x̄, W̄, b̄, nothing
   end
   y, cached_dense_adjoint
end
