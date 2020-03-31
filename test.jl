###
using BenchmarkTools
using CuArrays
using Flux

f(m, x) = m(x)

in, out, batchsize = 1536, 1024, 64
σ = Flux.leakyrelu
m = CachedDense(in, out, σ; batchsize=batchsize) |> Flux.gpu
mf = Flux.Dense(in, out, σ) |> Flux.gpu

x = rand(Float32, in) |> Flux.gpu;
@benchmark CuArrays.@sync f($m, $x)
@benchmark CuArrays.@sync f($mf, $x)
x = rand(Float32, in, batchsize÷2) |> Flux.gpu;
@benchmark CuArrays.@sync f($m, $x)
@benchmark CuArrays.@sync f($mf, $x)
x = rand(Float32, in, batchsize) |> Flux.gpu;
@benchmark CuArrays.@sync f($m, $x)
@benchmark CuArrays.@sync f($mf, $x)

x = rand(Float32, in) |> Flux.gpu;
CuArrays.@time f(m, x);
CuArrays.@time f(mf, x);
x = rand(Float32, in, batchsize÷2) |> Flux.gpu;
CuArrays.@time f(m, x);
CuArrays.@time f(mf, x);
x = rand(Float32, in, batchsize) |> Flux.gpu;
CuArrays.@time f(m, x);
CuArrays.@time f(mf, x);


cdims = Flux.NNlib.DenseConvDims(x,x)
@edit Flux.conv!(x, m.W, x,cdims)

###
x = rand(Float32, in, batchsize) |> gpu
m = CachedDense(in, out; batchsize=batchsize)  |> gpu
θ = Flux.params(m)
mf = Dense(copy(m.W), copy(m.b)) |> gpu
θf = Flux.params(mf)

mf(x) == m(x)

Juno.@profiler gs = gradient(θ) do
   sum(m(x))
end
@time gsf = gradient(θf) do
   sum(mf(x))
end

gsf[mf.W] == gs[m.W]
gsf[mf.b] == gs[m.b]

function reorder(x::AbstractMatrix)
    o = size(x,1) ÷ 4
    [x[1:(2o),:];
     x[(3o+1):end,:];
     x[(2o+1):(3o),:]]
end
function reorder(x::AbstractVector)
    o = size(x,1) ÷ 4
    [x[1:(2o)];
     x[(3o+1):end];
     x[(2o+1):(3o)]]
end

in, out, batchsize = 3, 5, 7
x = rand(Float32, in, batchsize)
m = Flux.LSTM(in, out); m(x)
(h, c) = m.state
m = CachedLSTMCell(in, out; batchsize=3batchsize)
mf = Flux.LSTMCell(reorder(m.Wi), reorder(m.Wh), reorder(m.b), zero(m.b), zero(m.b))

(h1, c1), h1 = m((copy(h), copy(c)), x)
(h2, c2), h2 = mf((copy(h), copy(c)), x)

h1 == h2
c1 == c2
