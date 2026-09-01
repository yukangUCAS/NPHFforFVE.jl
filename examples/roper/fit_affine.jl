# Affine NativeMinuit fit setup for the Roper example.
#
# The six regulator cutoffs are held fixed.  The remaining interaction
# couplings are exactly affine, while m_Roper_bare is a rest-frame dynamic
# mass and is updated only in the kinetic term.

using NPHFforFVE
using NativeMinuit
using Random

include(joinpath(@__DIR__, "project.jl"))
include(joinpath(@__DIR__, "potential_defs.jl"))

# Cutoffs and one selected linear coupling are fixed at their scenario-II
# values. prepare_affine absorbs all fixed fields into its constant term.
const affine_fixed = (
    g_Npi_Npi=true,
    Lambda_R_Npi=true,
    Lambda_R_Deltapi=true,
    Lambda_R_Nsigma=true,
    Lambda_Npi=true,
    Lambda_Deltapi=true,
    Lambda_Nsigma=true,
)

# Pseudo-data: the spectrum just computed with Ncuts = [0, 10, 10, 10].
const pseudo_data = [
    [1289.85, 1408.42, 1536.56,
     1587.04, 1644.85, 1778.83,
     1856.29, 1905.64],
    [1287.70, 1292.69, 1398.90,
     1439.00, 1514.27, 1543.32,
     1566.02, 1624.13],
]

# Use the same absolute pseudo-data uncertainty for every fitted level.
const pseudo_error = 5.0 # MeV
data = SpectrumDataset([
    SpectrumGroup(1, "G1+", pseudo_data[1]; errors=fill(pseudo_error, 8)),
    SpectrumGroup(2, "G1+", pseudo_data[2]; errors=fill(pseudo_error, 8)),
])

# Prepare the projected affine interaction. Fixed fields are absorbed into V0;
# the dynamic bare mass is updated in the kinetic term at each evaluation.
prepared = prepare_affine(
    project, my_V, RoperParams();
    backend=:projected_blocks,
    channel_filter=channel_filter,
    entry_filter=entry_filter,
    fixed=affine_fixed,
)

# Reproducible independent 10%--20% relative perturbations with random signs.
reference_params = RoperParams()
rng = MersenneTwister(20260830)

parameter_names = Symbol.(param_names(RoperParams))
x0 = to_vector(reference_params)
free_indices = findall(
    name -> !(name in keys(affine_fixed)), parameter_names)
for i in free_indices
    shift = 0.10 + 0.10 * rand(rng)
    rand(rng, Bool) && (shift = -shift)
    x0[i] *= 1.0 + shift
end

initial_params = from_vector(RoperParams, x0)
initial_steps = 0.01 .* abs.(x0)

problem = SpectrumFitProblem(prepared, initial_params, data)

m = minuit(problem; initial_steps=initial_steps, fixed=affine_fixed)
migrad!(m; iterate=1)

pbest = best_params(problem, m)
println("Roper affine fit")
println("  valid: ", m.valid)
println("  chi²: ", m.fval)
println("  function evaluations: ", m.nfcn)
println("  best-fit parameters: ", pbest)
