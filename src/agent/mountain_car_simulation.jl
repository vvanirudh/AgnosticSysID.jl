import Random: MersenneTwister

function evaluate(
    mountaincar::MountainCar,
    params::MountainCarParameters,
    horizon::Int64,
    num_evaluations::Int64;
    rng::MersenneTwister,
    policy = nothing,
)::Float64
    evaluation_cost = 0.0
    for _ = 1:num_evaluations
        evaluation_cost += cost_episode(
            simulate_episode(mountaincar, params, horizon; rng = rng, policy = policy),
        )
    end
    evaluation_cost / num_evaluations
end

function simulate_episode(
    mountaincar::MountainCar,
    params::MountainCarParameters,
    horizon::Int64;
    rng::MersenneTwister,
    policy::Array{Int64},
)::Array{MountainCarContTransition}
    episode_data = []
    cont_state = init(mountaincar; cont = true)
    actions = getActions(mountaincar)
    for t = 1:horizon
        action = actions[policy[cont_state_to_idx(mountaincar, cont_state)]]
        cont_state_next, cost = step(mountaincar, cont_state, action, params)
        push!(
            episode_data,
            MountainCarContTransition(cont_state, action, cost, cont_state_next),
        )
        if checkGoal(mountaincar, cont_state_next)
            break
        end
        cont_state = cont_state_next
    end
    episode_data
end

function cost_episode(episode::Array{MountainCarContTransition})::Float64
    cost_val = 0.0
    for transition in episode
        cost_val += transition.cost
    end
    cost_val
end

function random_policy(mountaincar::MountainCar; rng = nothing)
    n_states = mountaincar.position_discretization * mountaincar.speed_discretization
    n_actions = 2
    if isnothing(rng)
        rand(1:n_actions, n_states)
    else
        rand(rng, 1:n_actions, n_states)
    end
end
