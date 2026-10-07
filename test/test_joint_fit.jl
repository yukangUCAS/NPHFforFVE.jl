using NPHFforFVE
using Test

@params struct JointFitParams
    mass = 500.0
end

@params struct OtherJointFitParams
    mass = 500.0
end

@testset "joint fit across reference frames" begin
    initial = JointFitParams()
    channel = FockChannel(
        "scalar", [1], [:boson], [dynamic_mass(:mass)],
        [0//1], [0//1], [1.0], relativistic)
    potential = (args...) -> 0.0 + 0.0im
    problems = SpectrumFitProblem[]
    momentum_squared = (2π * NPHFforFVE.ħc / (24 * 0.1))^2

    # Both datasets use config 1, but belong to distinct reference frames.
    for (momentum, irrep, p_squared, error) in
        ((D000, "A1+", 0.0, 2.0), (D001, "A1", momentum_squared, 3.0))
        project = Project(momentum, 0//1, [channel], [0])
        add_config!(project, 24, 0.1, [irrep], [1])
        prepared = prepare_spectrum(project; backend=:complete_matrix)
        data = SpectrumDataset([
            SpectrumGroup(1, irrep, [sqrt(initial.mass^2 + p_squared)];
                          errors=[error]),
        ])
        push!(problems, SpectrumFitProblem(prepared, potential, initial, data))
    end

    joint = SpectrumFitProblem(problems)
    @test joint(to_vector(initial)) ≈ 0.0 atol=1e-10
    trial = JointFitParams(mass=525.0)
    trial_vector = to_vector(trial)
    expected = ((trial.mass - initial.mass) / 2)^2 +
               ((sqrt(trial.mass^2 + momentum_squared) -
                 sqrt(initial.mass^2 + momentum_squared)) / 3)^2
    @test joint(trial_vector) ≈ expected atol=1e-8
    @test joint(trial_vector) ≈ sum(problem(trial_vector) for problem in problems)
    @test joint.parameter_names == ["mass"]
    @test length(joint.evaluator(trial)) == 2

    @test_throws ArgumentError SpectrumFitProblem(SpectrumFitProblem[])
    first_problem = first(problems)
    for (params, names) in ((OtherJointFitParams(), ["mass"]),
                            (initial, ["other"]),
                            (trial, ["mass"]))
        incompatible = SpectrumFitProblem(
            first_problem.evaluator, params, first_problem.data,
            first_problem.loss, names)
        @test_throws ArgumentError SpectrumFitProblem([first_problem, incompatible])
    end
end
