using BenchmarkTools
using CuArrays

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
