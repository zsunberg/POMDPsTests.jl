

function generate_sor(s::Int, a::Int, rng::AbstractRNG)
    sp = s+a
    o = sp+randn(rng)
    r = sp^2
    return sp, o, r
end

function iter(s, a)
    sp, o, r = generate_sor(s, a, Base.GLOBAL_RNG)
    return sp+o+r
end

N = 1_000_000

s = 1
a = 1
r_sum = 0.0
o_sum = 0.0
@code_warntype iter(s,a)
for i in 1:N
    sp, o, r = generate_sor(s, a, Base.GLOBAL_RNG)
    r_sum += r
    o_sum += o
    s = sp
end

rng = MersenneTwister(1)
s = 1
a = 1
r_sum = 0.0
@time for i in 1:N
    sp, o, r = generate_sor(s, a, rng)
    r_sum += r
    o_sum += o
    s = sp
end
@show r_sum

rng = MersenneTwister(1)
s = 1
a = 1
r_sum = 0.0
@time for i in 1:N
    sp = s+a
    o = sp+randn(rng)
    r = sp^2
    r_sum += r
    o_sum += o
    s = sp
end
@show r_sum

rng = MersenneTwister(1)
s = 1
a = 1
r_sum = 0.0
@time for i in 1:N
    sp = s+a
    o = sp+randn(rng)
    r = sp^2
    r_sum += r
    o_sum += o
    s = sp
end
@show r_sum
