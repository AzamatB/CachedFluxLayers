using LinearAlgebra
import Flux
import Flux: σ

gate(h, n) = (1:h) .+ h*(n-1)
gate(x::AbstractVector, h, n) = @view x[gate(h,n)]
gate(x::AbstractMatrix, h, n) = @view x[gate(h,n),:]

struct CachedRecur{C, M <: AbstractVecOrMat}
   cell  :: C
   init  :: M
   state :: M
end

Flux.@functor CachedRecur

Flux.trainable(m::CachedRecur) = (m.cell, m.init)

struct CachedLSTMCell{M <: AbstractVecOrMat, V <: AbstractVecOrMat}
   Wi :: M
   Wh :: M
   b  :: V
   cache :: NamedTuple{(:g, :Wix), NTuple{2,M}}
end

Flux.@functor CachedLSTMCell

Flux.trainable(m::CachedLSTMCell) = (m.Wi, m.Wh, m.b)

function CachedLSTMCell(in::Integer, out::Integer; batchsize::Integer,
                        init = Flux.glorot_uniform)
   b = init(4out)
   b[gate(out, 2)] .= 1
   O = Flux.zeros(4out, batchsize)
   cache = (g = O, Wix = copy(O))
   return CachedLSTMCell(init(4out, in), init(4out, out), b, cache)
end

function (m::CachedLSTMCell)((h, c), x)
   Wi, Wh, b, o, batch = m.Wi, m.Wh, m.b, size(h, 1), axes(x, 2)
   @views g, Wix = m.cache.g[:,batch], m.cache.Wix[:,batch]
   g .= mul!(Wix, Wi, x) .+ mul!(g, Wh, h) .+ b
   @views ifo, cell = g[1:(3o),:], g[(3o+1):end,:]
   ifo .= σ.(ifo)
   cell .= tanh.(cell)
   input, forget, output = gate(ifo, o, 1), gate(ifo, o, 2), gate(ifo, o, 3)
   @. c = forget * c + input * cell
   @. h = output * tanh(c)
   return (h, c), h
end
