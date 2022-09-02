import Plots: plot, plot!, xlabel!, ylabel!

function mountain_car_finite_agnostic_sys_id_experiment(
    rock_c::Float64,
    position_sigma::Float64;
    num_iterations::Int64 = 10,
    num_evaluations::Int64 = 1,
    seed::Int64 = 0,
)

    mountaincar = MountainCar(rock_c; position_sigma = position_sigma)
    model = MountainCar(0.0)
    horizon = 500
    agent = MountainCarFiniteAgnosticSysIDAgent(mountaincar, model, horizon)

    self_consistent_costs, optimal_cost = run(
        agent;
        args = create_self_consistent_loss_args(num_iterations, num_evaluations),
        seed = seed,
    )

    optimal_costs, _ = run(
        agent;
        args = create_optimal_loss_args(num_iterations, num_evaluations),
        seed = seed,
    )

    psuedo_costs, _ = run(
        agent;
        args = create_psuedo_loss_args(num_iterations, num_evaluations),
        seed = seed,
    )

    psuedo_badness_costs, _ = run(
        agent;
        args = create_psuedo_badness_loss_args(num_iterations, num_evaluations),
        seed = seed,
    )

    mle_costs, _ = run(
        agent;
        args = create_mle_loss_args(num_iterations, num_evaluations),
        seed = seed,
    )

    plot(1:num_iterations, self_consistent_costs, lw = 3, label = "Consistent-MA")
    plot!(1:num_iterations, ones(num_iterations) * optimal_cost, lw = 3, label = "Opt")
    plot!(1:num_iterations, optimal_costs, lw=3, label="Optimal-MA")
    plot!(1:num_iterations, psuedo_costs, lw=3, label="Psuedo-MA")
    plot!(1:num_iterations, psuedo_badness_costs, lw=3, label="Psuedo-Badness-MA")
    plot!(1:num_iterations, mle_costs, lw=3, label="MLE", legend=:bottomleft)
    xlabel!("Iterations")
    ylabel!("Cost")

end
