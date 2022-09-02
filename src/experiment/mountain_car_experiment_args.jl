struct MountainCarExperimentArgs
    self_consistent_loss::Bool
    optimal_loss::Bool
    psuedo_loss::Bool
    psuedo_badness_loss::Bool
    mixture_policy_data_collection::Bool
    optimal_policy_data_collection::Bool
    num_iterations::Int64
    num_evaluations::Int64
end

function create_self_consistent_loss_args(num_iterations::Int64, num_evaluations::Int64)
    MountainCarExperimentArgs(
        true, false, false, false, true, false,
        num_iterations, num_evaluations,
    )
end

function create_optimal_loss_args(num_iterations::Int64, num_evaluations::Int64)
    MountainCarExperimentArgs(
        true, true, false, false, true, true,
        num_iterations, num_evaluations,
    )
end

function create_psuedo_loss_args(num_iterations::Int64, num_evaluations::Int64)
    MountainCarExperimentArgs(
        true, false, true, false, true, true,
        num_iterations, num_evaluations,
    )
end

function create_mle_loss_args(num_iterations::Int64, num_evaluations::Int64)
    MountainCarExperimentArgs(
        false, false, false, false, true, false,
        num_iterations, num_evaluations,
    )
end

function create_psuedo_badness_loss_args(num_iterations::Int64, num_evaluations::Int64)
    MountainCarExperimentArgs(
        true, false, false, true, true, true,
        num_iterations, num_evaluations,
    )
end