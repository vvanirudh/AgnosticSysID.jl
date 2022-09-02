function iterative_policy_evaluation(
    mountaincar::MountainCar,
    params::Array{Float64},
    policy::Array{Int64};
    threshold = 1e-5,
    gamma = 1.0,
    max_iterations = 1e3,
)
    n_states = mountaincar.position_discretization * mountaincar.speed_discretization + 1
    T = generate_transition_matrix(mountaincar, params)
    R = generate_cost_vector(mountaincar)
    V_old = copy(R)
    V = 2 * V_old
    error_vec = get_error_vec(V, V_old)
    criterion = maximum(error_vec)
    count = 0
    while criterion >= threshold && count < max_iterations
        count += 1
        V_old = copy(V)
        next = [T[s, policy[s]] for s = 1:n_states-1]
        push!(next, n_states)  # absorbing state
        V = R .+ gamma .* V_old[next]
        error_vec = get_error_vec(V, V_old)
        criterion = maximum(error_vec)
    end
    V[1:n_states-1]
end
