# On Zach's machine, this outputs
#=
zach@Theresa:~/Desktop$ julia vi_allocation_test.jl 
 51.437988 seconds (85.34 M allocations: 2.685 GB, 0.51% gc time)
 53.665579 seconds (74.02 M allocations: 2.062 GB, 0.39% gc time)
=#

using POMDPModels
using POMDPs
using DiscreteValueIteration
using POMDPToolbox

function solve_no_allocation{S,A}(solver::ValueIterationSolver, mdp::Union{MDP{S,A},POMDP{S,A}}, policy=create_policy(solver, mdp); verbose::Bool=false)

    # solver parameters
    max_iterations = solver.max_iterations
    belres = solver.belres
    discount_factor = discount(mdp)

    # intialize the utility and Q-matrix
    util = policy.util
    qmat = policy.qmat
    include_Q = policy.include_Q
    pol = policy.policy 

    # pre-allocate the transtion distirbution and the interpolants
    dist = create_transition_distribution(mdp)

    total_time = 0.0
    iter_time = 0.0

    # create an ordered list of states for fast iteration
    states = ordered_states(mdp)

    # main loop
    for i = 1:max_iterations
        residual = 0.0
        tic()
        # state loop
        for (istate,s) in enumerate(states)
            old_util = util[istate] # for residual 
            max_util = -Inf
            # action loop
            # util(s) = max_a( R(s,a) + discount_factor * sum(T(s'|s,a)util(s') )
            for (iaction, a) in enumerate(policy.action_map)
                dist = transition(mdp, s, a) # fills distribution over neighbors
                u = 0.0
                for sp in iterator(dist)
                    p = pdf(dist, sp)
                    p == 0.0 ? continue : nothing # skip if zero prob
                    r = reward(mdp, s, a, sp)
                    sidx = state_index(mdp, sp)
                    u += p * (r + discount_factor * util[sidx]) 
                end
                new_util = u 
                if new_util > max_util
                    max_util = new_util
                    pol[istate] = iaction
                end
                include_Q ? (qmat[istate, iaction] = new_util) : nothing
            end # action
            # update the value array
            util[istate] = max_util 
            diff = abs(max_util - old_util)
            diff > residual ? (residual = diff) : nothing
        end # state
        iter_time = toq()
        total_time += iter_time
        verbose ? println("Iteration : $i, residual: $residual, iteration run-time: $iter_time, total run-time: $total_time") : nothing
        residual < belres ? break : nothing
    end # main
    policy
end



function solve_pre_allocation{S,A}(solver::ValueIterationSolver, mdp::Union{MDP{S,A},POMDP{S,A}}, policy=create_policy(solver, mdp); verbose::Bool=false)

    # solver parameters
    max_iterations = solver.max_iterations
    belres = solver.belres
    discount_factor = discount(mdp)

    # intialize the utility and Q-matrix
    util = policy.util
    qmat = policy.qmat
    include_Q = policy.include_Q
    pol = policy.policy 

    # pre-allocate the transtion distirbution and the interpolants
    dist = create_transition_distribution(mdp)

    total_time = 0.0
    iter_time = 0.0

    # create an ordered list of states for fast iteration
    states = ordered_states(mdp)

    # main loop
    for i = 1:max_iterations
        residual = 0.0
        tic()
        # state loop
        for (istate,s) in enumerate(states)
            old_util = util[istate] # for residual 
            max_util = -Inf
            # action loop
            # util(s) = max_a( R(s,a) + discount_factor * sum(T(s'|s,a)util(s') )
            for (iaction, a) in enumerate(policy.action_map)
                dist = transition(mdp, s, a, dist) # fills distribution over neighbors
                u = 0.0
                for sp in iterator(dist)
                    p = pdf(dist, sp)
                    p == 0.0 ? continue : nothing # skip if zero prob
                    r = reward(mdp, s, a, sp)
                    sidx = state_index(mdp, sp)
                    u += p * (r + discount_factor * util[sidx]) 
                end
                new_util = u 
                if new_util > max_util
                    max_util = new_util
                    pol[istate] = iaction
                end
                include_Q ? (qmat[istate, iaction] = new_util) : nothing
            end # action
            # update the value array
            util[istate] = max_util 
            diff = abs(max_util - old_util)
            diff > residual ? (residual = diff) : nothing
        end # state
        iter_time = toq()
        total_time += iter_time
        verbose ? println("Iteration : $i, residual: $residual, iteration run-time: $iter_time, total run-time: $total_time") : nothing
        residual < belres ? break : nothing
    end # main
    policy
end

mdp = GridWorld()
solver = ValueIterationSolver()
solve_no_allocation(solver, mdp, ValueIterationPolicy(mdp, include_Q=false))
@time for i in 1:100
    solve_no_allocation(solver, mdp, ValueIterationPolicy(mdp, include_Q=false))
end

mdp = GridWorld()
solver = ValueIterationSolver()
solve_pre_allocation(solver, mdp, ValueIterationPolicy(mdp, include_Q=false))
@time for i in 1:100
    solve_pre_allocation(solver, mdp, ValueIterationPolicy(mdp, include_Q=false))
end
