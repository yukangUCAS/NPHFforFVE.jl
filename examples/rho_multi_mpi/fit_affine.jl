# Joint rest- and moving-frame multi-m_pi fit. Set NPHF_RUN_FIT=1 to fit.
using NPHFforFVE

include(joinpath(@__DIR__, "potential_defs.jl"))
include(joinpath(@__DIR__, "potential_moving_defs.jl"))

# Rest-frame rho-pipi project; masses are resolved per config.
rho = FockChannel(
    "rho", [1], [:boson], [dynamic_mass(:m_rho)],
    [1//1], [1//1], [-1.0], relativistic)
pipi = FockChannel(
    "pipi", [2], [:boson], [mass_unfixed(:m_pi)],
    [0//1], [1//1], [1.0], relativistic)

project = Project(D000, 1//1, [rho, pipi], [0, 15])
add_config!(project, 32, 0.1, ["T1-"], [5]; m_pi=208.0)
add_config!(project, 48, 0.1, ["T1-"], [5]; m_pi=305.0)

moving_project = Project(D001, 1//1, [rho, pipi], [0, 15])
add_config!(moving_project, 32, 0.1, ["A1"], [2]; m_pi=208.0)

# Each file uses config indices local to its own project.
function read_pseudodata(filename, n_configs)
    energies = [Float64[] for _ in 1:n_configs]
    for line in eachline(joinpath(@__DIR__, filename))
        (isempty(line) || startswith(line, '#') || startswith(line, "config\t")) && continue
        columns = split(line, '\t')
        config_index = parse(Int, columns[1])
        level = parse(Int, columns[7])
        level == length(energies[config_index]) + 1 ||
            error("$filename levels must be ordered within each config")
        push!(energies[config_index], parse(Float64, columns[8]))
    end
    return energies
end
energies = read_pseudodata("pseudodata.tsv", length(project.configs))
moving_energies = read_pseudodata("pseudodata_moving.tsv", length(moving_project.configs))

# The TSV has no uncertainties; assign 0.5 MeV to each pseudo-data level.
pseudo_error = 0.5  # MeV
data = SpectrumDataset([
    SpectrumGroup(i, "T1-", energies[i]; errors=fill(pseudo_error, 5))
    for i in eachindex(energies)
])
moving_data = SpectrumDataset([
    SpectrumGroup(1, "A1", moving_energies[1]; errors=fill(pseudo_error, 2)),
])

@params struct FitParams
    c0 = 720.0
    c1 = 0.0
    g_208 = 1.7e-5
    g_305 = 2.7e-5
    h = 0.1e-5
end

# Bare rho mass: m_rho = c0 + c1 * m_pi^2.
# The two g values are independent; h is shared by both configs.
g_Dict = Dict(208.0 => :g_208, 305.0 => :g_305)
dependence = ParaMpiDependence(
    shared_fit=(:h,),
    dependent_fit=(
        m_rho=depends_on((:c0, :c1),
            (config, fit_params) -> fit_params.c0 + fit_params.c1 * config.m_pi^2),
        g=depends_on((:g_208, :g_305),
            (config, fit_params) -> getproperty(fit_params, g_Dict[config.m_pi])),
    ),
    known=(m_pi=known_from(config -> config.m_pi),),
    fixed=(cutoff=900.0,),
)

initial_val = FitParams()
pmap = ParamMapping(RhoPipiParams, dependence, initial_val)
prepared = prepare_affine(project, my_V, pmap;
                          backend=:projected_blocks,
                          channel_filter=channel_filter)
problem_rest = SpectrumFitProblem(prepared, initial_val, data)

# Masses enter the moving-frame boost, so use general spectrum preparation.
prepared_moving = prepare_spectrum(moving_project; backend=:projected_blocks)
problem_moving = SpectrumFitProblem(prepared_moving, my_V_moving, pmap, moving_data)
problem = SpectrumFitProblem([problem_rest, problem_moving])

run_fit = get(ENV, "NPHF_RUN_FIT", "0") == "1"
if run_fit
    using NativeMinuit

    initial_steps = 0.05 .* abs.(to_vector(initial_val))
    initial_steps[2] = max(initial_steps[2], 1e-4)  # c1 may start at zero
    fit = minuit(problem;
                 initial_steps=initial_steps,
                 limits=(g_208=(0.0, Inf), g_305=(0.0, Inf)))
    migrad!(fit; maxfcn=3000)

    fitted_params = best_params(problem, fit)
    fitted_spectra = problem.evaluator(fitted_params)
    println("valid: ", fit.valid)
    println("chi²: ", fit.fval)
    println("best-fit parameters: ", fitted_params)
    for config_index in eachindex(project.configs)
        println("rest config ", config_index, " T1-: ", fitted_spectra[1][config_index]["T1-"])
    end
    println("moving config 1 A1: ", fitted_spectra[2][1]["A1"])
end
