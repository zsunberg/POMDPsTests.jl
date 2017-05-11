N = 10_000
rngs = Array(MersenneTwister, N)
rngs2 = Array(MersenneTwister, N)

for i in 1:N
    rngs[i] = MersenneTwister(i)
end

for i in 1:N
    srand(rngs[i], i)
end

for i in 1:N
    rngs2[i] = copy(rngs[i])
end

for i in 1:N
    copy!(rngs2[i], rngs[i])
end

println("create new")
@time for i in 1:N
    rngs[i] = MersenneTwister(i)
end

println("srand")
@time for i in 1:N
    srand(rngs[i], i)
end

println("copy")
@time for i in 1:N
    rngs2[i] = copy(rngs[i])
end

println("copy!")
@time for i in 1:N
    copy!(rngs2[i], rngs[i])
end

