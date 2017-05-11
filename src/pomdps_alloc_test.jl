using POMDPs
import POMDPs: create_state, discount, reward
using POMDPToolbox

type ImageMDP <: MDP{Matrix{Int},Int}
    size::Tuple{Int, Int}
end

create_state(mdp::ImageMDP) = Array(Int, mdp.size[1], mdp.size[2])
discount(::ImageMDP) = 0.9
function generate_sr(mdp::ImageMDP, s::Matrix{Int}, a::Int, rng::AbstractRNG, sp::Matrix{Int})
    copy!(sp, s)
    r = sp[a,a]
    i = rand(rng, 1:mdp.size[1])
    j = rand(rng, 1:mdp.size[2])
    sp[i,j] += a
    return sp, r
end
function generate_sr(mdp::ImageMDP, s::Matrix{Int}, a::Int, rng::AbstractRNG)
    sp = copy(s)
    r = sp[a,a]
    i = rand(rng, 1:mdp.size[1])
    j = rand(rng, 1:mdp.size[2])
    sp[i,j] += a
    return sp, r
end
function step!(s::Matrix{Int}, mdp::ImageMDP, a::Int, rng::AbstractRNG)
    i = rand(rng, 1:mdp.size[1])
    j = rand(rng, 1:mdp.size[2])
    s[i,j] += a
    return s
end
function reward(mdp::ImageMDP,s::Matrix{Int},a::Int)
    return s[a,a]
end

type IPolicy <: Policy{Matrix{Int}}
    addend::Int
end
action(p::IPolicy, s::Matrix{Int}, a::Int=1) = p.addend


function simulate_no_alloc{S,A}(sim::RolloutSimulator, mdp::MDP{S,A}, policy::Policy, initial_state::S)

    eps = get(sim.eps, 0.0)
    max_steps = get(sim.max_steps, typemax(Int))

    s = initial_state

    disc = 1.0
    r_total = 0.0
    step = 1

    while disc > eps && !isterminal(mdp, s) && step <= max_steps
        a = action(policy, s)

        sp, r = generate_sr(mdp, s, a, sim.rng)

        r_total += disc*r

        s = sp

        disc *= discount(mdp)
        step += 1
    end

    return r_total
end


function simulate_alloc{S,A}(sim::RolloutSimulator, mdp::MDP{S,A}, policy::Policy, initial_state::S)

    eps = get(sim.eps, 0.0)
    max_steps = get(sim.max_steps, typemax(Int))

    s = deepcopy(initial_state)
    sp = create_state(mdp)

    a = create_action(mdp)

    disc = 1.0
    r_total = 0.0
    step = 1

    while disc > eps && !isterminal(mdp, s) && step <= max_steps
        a = action(policy, s, a)

        sp, r = generate_sr(mdp, s, a, sim.rng, sp)

        r_total += disc*r

        # alternates using the memory allocated for s and sp so nothing new has to be allocated
        tmp = s
        s = sp
        sp = tmp

        disc *= discount(mdp)
        step += 1
    end

    return r_total
end

function simulate_step{S,A}(sim::RolloutSimulator, mdp::MDP{S,A}, policy::Policy, initial_state::S)

    eps = get(sim.eps, 0.0)
    max_steps = get(sim.max_steps, typemax(Int))

    s = deepcopy(initial_state)
    sp = create_state(mdp)

    a = create_action(mdp)

    disc = 1.0
    r_total = 0.0
    step = 1

    while disc > eps && !isterminal(mdp, s) && step <= max_steps
        a = action(policy, s, a)

        r = reward(mdp, s, a)
        step!(s, mdp, a, rng)

        r_total += disc*r

        disc *= discount(mdp)
        step += 1
    end

    return r_total
end

mdp = ImageMDP((500,500))
policy = IPolicy(147)

rng = MersenneTwister(123)
is = rand(rng, Int, mdp.size...)
@show simulate_alloc(RolloutSimulator(max_steps=500, rng=rng), mdp, policy, is)
rng = MersenneTwister(123)
is = rand(rng, Int, mdp.size...)
@show simulate_no_alloc(RolloutSimulator(max_steps=500, rng=rng), mdp, policy, is)
rng = MersenneTwister(123)
is = rand(rng, Int, mdp.size...)
@show simulate_step(RolloutSimulator(max_steps=500, rng=rng), mdp, policy, is)


println("Step")
rtot=0.0
@time for i = 1:100
    rng = MersenneTwister(i)
    is = rand(rng, Int, mdp.size...)
    rtot += simulate_step(RolloutSimulator(max_steps=500, rng=rng), mdp, policy, is)
end
@show rtot

println("Pre Allocation:")
rtot=0.0
@time for i = 1:100
    rng = MersenneTwister(i)
    is = rand(rng, Int, mdp.size...)
    rtot += simulate_alloc(RolloutSimulator(max_steps=500, rng=rng), mdp, policy, is)
end
@show rtot

println("No Allocation")
rtot=0.0
@time for i = 1:100
    rng = MersenneTwister(i)
    is = rand(rng, Int, mdp.size...)
    rtot += simulate_no_alloc(RolloutSimulator(max_steps=500, rng=rng), mdp, policy, is)
end
@show rtot
