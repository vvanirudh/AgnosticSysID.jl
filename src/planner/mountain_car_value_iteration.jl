function generate_transition_matrix(mountaincar::MountainCar, params::Array{Float64})
    n_states = mountaincar.position_discretization * mountaincar.speed_discretization + 1
    n_actions = 2
    T = zeros(Int64, (n_states, n_actions))
    actions = getActions(mountaincar)
    for a = 1:n_actions
        u = actions[a]
        for s = 1:n_states-1
            disc_state = idx_to_disc_state(mountaincar, s)
            if checkGoal(mountaincar, disc_state)
                T[s, a] = absorbing_state_idx(mountaincar)  # absorbing state
            else
                s_next = disc_state_to_idx(
                    mountaincar,
                    step(mountaincar, disc_state, u, params)[1],
                )
                T[s, a] = s_next
            end
        end
        T[n_states, a] = n_states  # absorbing state
    end
    T
end

function generate_cost_vector(mountaincar::MountainCar)
    n_states = mountaincar.position_discretization * mountaincar.speed_discretization + 1
    R = zeros(n_states)
    for s = 1:n_states-1
        R[s] = getCost(mountaincar, idx_to_disc_state(mountaincar, s))
    end
    R[n_states] = 0
    R
end

function get_error_vec(V::Array{Float64}, V_old::Array{Float64})::Array{Float64}
    # abs.(V - V_old) ./ (V_old .+ 1e-50)
    abs.(V - V_old)
end

function value_iteration(mountaincar::MountainCar, params::MountainCarParameters)
    value_iteration(mountaincar, vec(params))
end

function value_iteration(mountaincar::MountainCar, params::Array{Float64})
    T = generate_transition_matrix(mountaincar, params)
    R = generate_cost_vector(mountaincar)
    value_iteration(mountaincar, T, R)
end

function value_iteration(
    mountaincar::MountainCar,
    transition_matrix::Matrix{Int64},
    cost_vector::Vector{Float64};
    threshold = 1e-5,
    gamma = 1.0,
    max_iterations = 1e3,
)
    n_states = mountaincar.position_discretization * mountaincar.speed_discretization + 1
    n_actions = 2
    V_old = copy(cost_vector)
    pi = zeros(Int64, n_states)
    V = 2 * V_old
    Q = zeros((n_states, n_actions))
    error_vec = get_error_vec(V, V_old)
    criterion = maximum(error_vec)
    count = 0
    while criterion >= threshold && count < max_iterations
        count += 1
        V_old = copy(V)
        for a = 1:n_actions
            Q[:, a] = cost_vector .+ gamma .* V_old[transition_matrix[:, a]]
        end
        idxs = argmin(Q, dims = 2)
        for s = 1:n_states
            pi[s] = idxs[s][2]
        end
        V = Q[idxs]
        error_vec = get_error_vec(V, V_old)
        criterion = maximum(error_vec)
    end
    if count == max_iterations
        @warn "Value iteration exceeded max iterations"
    end
    # removing absorbing state
    pi[1:n_states-1], V[1:n_states-1], count != max_iterations
end
