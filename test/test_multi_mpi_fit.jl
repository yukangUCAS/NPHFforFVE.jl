using NPHFforFVE
using Test

@params struct MultiMpiInteractionParams
    mass = 400.0
    m_pi = 140.0
    g = 1.0
    h = 0.2
    cutoff = 900.0
end

@params struct MultiMpiFitParams
    c0 = 400.0
    c1 = 0.001
    g_140 = 1.0
    g_220 = 2.0
    h = 0.2
end

@testset "multi-m_pi affine fit path" begin
    channel = FockChannel(
        "scalar", [1], [:boson], [mass_unfixed(:mass)],
        [0//1], [0//1], [1.0], relativistic)
    project = Project(D000, 0//1, [channel], [0])
    add_config!(project, 24, 0.1, ["A1+"], [1]; m_pi=140.0)
    add_config!(project, 24, 0.1, ["A1+"], [1]; m_pi=220.0)

    potential = function (args...)
        p = args[end]
        return complex((p.g + p.h) * (1 + p.m_pi / 1000) /
                       (1 + (p.cutoff / 1000)^2))
    end
    initial = MultiMpiFitParams()
    g_field = Dict(140.0 => :g_140, 220.0 => :g_220)
    dependence = ParaMpiDependence(
        shared_fit=(:h,),
        dependent_fit=(
            mass=depends_on((:c0, :c1),
                (config, p) -> p.c0 + p.c1 * config.m_pi^2),
            g=depends_on((:g_140, :g_220),
                (config, p) -> getproperty(p, g_field[config.m_pi])),
        ),
        known=(m_pi=known_from(config -> config.m_pi),),
        fixed=(cutoff=900.0,),
    )
    pmap = ParamMapping(MultiMpiInteractionParams, dependence, initial)
    @test resolve_params(pmap, project.configs[1], initial).mass ≈ 419.6
    @test resolve_params(pmap, project.configs[2], initial).g == 2.0

    prepared = prepare_affine(project, potential, pmap;
                              backend=:projected_blocks)
    direct = prepare_spectrum(project; backend=:projected_blocks)
    for fit_params in (initial,
                       MultiMpiFitParams(c0=425.0, g_140=1.5, g_220=2.5))
        cached_result = compute!(prepared, fit_params)
        direct_result = compute!(direct, potential, pmap, fit_params)
        for config_index in 1:2
            @test cached_result[config_index]["A1+"] ≈
                  direct_result[config_index]["A1+"] atol=1e-8
        end
    end

    truth = compute!(prepared, initial)
    data = SpectrumDataset([
        SpectrumGroup(i, "A1+", truth[i]["A1+"]; errors=[1.0])
        for i in 1:2
    ])
    problem = SpectrumFitProblem(prepared, initial, data)
    @test problem(to_vector(initial)) ≈ 0.0 atol=1e-10
    @test problem(to_vector(MultiMpiFitParams(c0=425.0))) > 0
end
