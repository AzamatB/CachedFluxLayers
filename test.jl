using BenchmarkTools
f(m, x) = m(x)

m = Dense(900, 1200, Flux.leakyrelu; batchsize=700)
mf = Flux.Dense(900, 1200, Flux.leakyrelu)

x = rand(Float32, 900);
@benchmark f($m, $x)
@benchmark f($mf, $x)
x = rand(Float32, 900, 500);
@benchmark f($m, $x)
@benchmark f($mf, $x)
x = rand(Float32, 900, 700);
@benchmark f($m, $x)
@benchmark f($mf, $x)

cdims = Flux.NNlib.DenseConvDims(x,x)
@edit Flux.conv!(x, m.W, x,cdims)
