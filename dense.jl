using LinearAlgebra

import Flux
import ZygoteRules

struct CachedDense{M <: AbstractVecOrMat, V <: AbstractVecOrMat, F}
   W :: M
   b :: V
   σ :: F
   cache :: NamedTuple{(:y, :x̄, :W̄, :b̄), NTuple{4,M}}
end

Flux.trainable(m::CachedDense) = (m.W, m.b)

Flux.@functor CachedDense

CachedDense(W, b, cache) = CachedDense(W, b, identity, cache)

function CachedDense(in::Integer, out::Integer, σ = identity; batchsize::Integer,
               initW = Flux.glorot_uniform, initb = Flux.zeros)
   cache = (y = Flux.zeros(out, batchsize), x̄ = Flux.zeros(in, batchsize), W̄ = Flux.zeros(out, in), b̄ = Flux.zeros(out, batchsize))
   return CachedDense(initW(out, in), initb(out), σ, cache)
end

function Base.show(io::IO, m::CachedDense)
   out, in = size(m.W)
   batchsize = size(m.cache.y, 2)
   print(io, "CachedDense($in, $out")
   m.σ == identity || print(io, ", ", m.σ)
   print(io, "; batchsize=$batchsize)")
end

function (m::CachedDense)(x::AbstractVecOrMat)
   W, b, σ, c = m.W, m.b, m.σ, m.cache
   y = Wx = @view c.y[:,axes(x,2)]
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

ZygoteRules.@adjoint function (m::CachedDense)(x::AbstractVecOrMat, θ)
   W, b, σ, c = m.W, m.b, m.σ, m.cache
   batch, W̄ = axes(x, 2), c.W̄
   @views y, x̄, Wx = c.y[:,batch], c.x̄[:,batch], c.b̄[:,batch]
   Wx⁺b = W̄x̄⁺b̄ = Wx
   mul!(Wx, W, x)
   @. Wx⁺b = Wx + b
   @. y = σ(Wx⁺b)
   σ′ = σ'
   function CachedDense_adjoint(ȳ)
      @. W̄x̄⁺b̄ = ȳ * σ′(Wx⁺b)
      mul!(x̄, W', W̄x̄⁺b̄)
      mul!(W̄, W̄x̄⁺b̄, x')
      b̄ = W̄x̄⁺b̄
      return x̄, W̄, b̄
   end
   y, CachedDense_adjoint
end
