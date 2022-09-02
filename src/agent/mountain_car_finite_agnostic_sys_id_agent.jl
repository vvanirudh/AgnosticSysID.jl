import Random: MersenneTwister

const true_params = MountainCarParameters(-0.0025, 3)

struct MountainCarFiniteAgnosticSysIDAgent
    mountaincar::MountainCar
    model::MountainCar
    model_class::Array{Tuple{MountainCar,MountainCarParameters}}
    horizon::Int64
end

function MountainCarFiniteAgnosticSysIDAgent(
    mountaincar::MountainCar,
    model::MountainCar,
    horizon::Int64,
)
    model_class = generate_model_class(model)
    MountainCarFiniteAgnosticSysIDAgent(mountaincar, model, model_class, horizon)
end

function generate_model_class(
    model::MountainCar,
)::Array{Tuple{MountainCar,MountainCarParameters}}
    theta1s = -0.00245:-0.000001:-0.00305
    theta2s = 2.9:0.001:3.1
    models = []
    for theta1 in theta1s
        for theta2 in theta2s
            push!(models, (model, MountainCarParameters(theta1, theta2)))
        end
    end
    models
end

function run(
    agent::MountainCarFiniteAgnosticSysIDAgent;
    args::MountainCarExperimentArgs,
    seed::Int64 = 0,
)
    rng = MersenneTwister(seed)
    # Choose an initial model
    model, params = agent.model_class[1]
    # Choose an initial policy
    policy, values, _ = value_iteration(model, params)
    # Evaluate optimal policy, values, and costs
    optimal_policy, optimal_values, _ = value_iteration(agent.mountaincar, true_params)
    optimal_cost = evaluate(
        agent.mountaincar,
        true_params,
        agent.horizon,
        args.num_evaluations;
        policy = optimal_policy,
        rng = rng,
    )
    # Initialize dataset
    on_policy_transitions::Array{Array{MountainCarContTransition}} = []
    off_policy_transitions::Array{Array{MountainCarContTransition}} = []
    # Initialize value functions
    all_values::Array{Array{Float64}} = []
    # Initialize losses
    losses = [0.0 for _ = 1:length(agent.model_class)]
    # All costs 
    total_costs::Array{Float64} = []

    # Start iterations
    n = args.num_iterations  # number of iterations
    m = 1  # number of rollouts using current policy
    p = 0  # number of rollouts using random/optimal policy
    if args.mixture_policy_data_collection
        p = 1
        rollout_policy = random_policy(agent.model; rng=rng)
        if args.optimal_policy_data_collection
            rollout_policy = optimal_policy
        end
    end
    for i = 1:n
        println("Running iteration ", i)
        push!(on_policy_transitions, [])
        push!(off_policy_transitions, [])
        # Do rollouts in the real world using policy to collect data
        total_cost = 0.0
        for _ = 1:m
            episode = simulate_episode(
                agent.mountaincar,
                true_params,
                agent.horizon;
                policy = rollout_policy,
                rng = rng,
            )
            on_policy_transitions[i] = vcat(on_policy_transitions[i], episode)
        end
        total_cost = evaluate(
            agent.mountaincar,
            true_params,
            agent.horizon,
            args.num_evaluations;
            policy = policy,
            rng = rng,
        )
        push!(total_costs, total_cost)
        # Do rollouts in the real world using random/optimal policy to collect data
        for _ = 1:p
            episode =
                simulate_episode(agent.mountaincar, true_params, agent.horizon; rng = rng, policy=rollout_policy)
            off_policy_transitions[i] = vcat(off_policy_transitions[i], episode)
        end
        # Store value function
        push!(all_values, values)
        if args.optimal_loss || args.psuedo_badness_loss
            opt_policy_values = iterative_policy_evaluation(model, vec(params), optimal_policy)
        end
        # Collected data, now update model
        for j = 1:length(agent.model_class)
            # Compute loss
            loss = 0.0
            if args.self_consistent_loss
                loss = compute_model_advantage_loss(
                    on_policy_transitions[i],
                    all_values[i],
                    agent.model_class[j],
                )
                if args.optimal_loss && args.optimal_policy_data_collection
                    loss += compute_model_advantage_loss(
                        off_policy_transitions[i],
                        opt_policy_values,
                        agent.model_class[j],
                    )
                elseif args.psuedo_loss && args.optimal_policy_data_collection
                    loss += compute_model_advantage_loss(
                        off_policy_transitions[i],
                        all_values[i],
                        agent.model_class[j],
                    )
                elseif args.psuedo_badness_loss && args.optimal_policy_data_collection
                    loss += compute_model_advantage_loss(
                        off_policy_transitions[i],
                        all_values[i],
                        agent.model_class[j],
                    )
                    loss += compute_badness_loss(
                        off_policy_transitions[i],
                        opt_policy_values,
                        all_values[i],
                        agent.model_class[j],
                    )
                elseif args.mixture_policy_data_collection
                    loss += compute_model_advantage_loss(
                        off_policy_transitions[i],
                        all_values[i],
                        agent.model_class[j],
                    )
                end
            else
                loss = compute_l2_loss(on_policy_transitions[i], agent.model_class[j])
                if args.mixture_policy_data_collection
                    loss += compute_l2_loss(off_policy_transitions[i], agent.model_class[j])
                end
            end
            loss = loss / (m + p)
            losses[j] += loss
        end
        # Find best model
        # TODO: Doing FTL now, but need to do better
        model, params = agent.model_class[argmin(losses)]
        println(
            "Best model so far has params ",
            vec(params),
            " and loss ",
            minimum(losses) / i,
        )
        # Compute corresponding policy and values
        policy, values, _ = value_iteration(model, params)
    end
    total_costs, optimal_cost
end

function compute_model_advantage_loss(
    transitions::Array{MountainCarContTransition},
    values::Array{Float64},
    model_params::Tuple{MountainCar,MountainCarParameters},
)
    # _, values, _ = value_iteration(agent.mountaincar, true_params)
    loss = 0.0
    model, params = model_params
    for transition in transitions
        predicted_state, _ =
            step(model, transition.initial_state, transition.action, params)
        loss += abs(
            values[cont_state_to_idx(model, transition.final_state)] -
            values[cont_state_to_idx(model, predicted_state)],
        )
    end
    loss
end

function compute_l2_loss(
    transitions::Array{MountainCarContTransition},
    model_params::Tuple{MountainCar,MountainCarParameters},
)
    model, params = model_params
    loss = 0.0
    for transition in transitions
        predicted_state, _ =
            step(model, transition.initial_state, transition.action, params)
        loss +=
            (predicted_state.position - transition.final_state.position)^2 +
            (predicted_state.speed - transition.final_state.speed)^2
    end
    loss
end

function compute_badness_loss(
    transitions::Array{MountainCarContTransition},
    opt_policy_values::Array{Float64},
    model_values::Array{Float64},
    model_params::Tuple{MountainCar, MountainCarParameters},
)
    loss = 0.0
    model, params = model_params
    for transition in transitions
        predicted_state, _ = step(model, transition.initial_state, transition.action, params)
        loss += abs(opt_policy_values[cont_state_to_idx(model, predicted_state)] - model_values[cont_state_to_idx(model, predicted_state)])
    end
    loss
end