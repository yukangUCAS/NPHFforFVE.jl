using Test

# Public regression suite: each file constructs a concrete physical system and
# checks its projected finite-volume spectrum.  Every file is loaded in an
# independent module because some systems intentionally reuse short names for
# masses, momenta, and symmetry groups.
const PHYSICS_TESTS = [
    "test_pipi.jl",
    "test_pipi_moving.jl",
    "test_piN.jl",
    "test_piN_moving.jl",
    "test_nn.jl",
    "test_nn_moving.jl",
    "test_rhoN.jl",
    "test_rhoN_moving.jl",
    "test_3pi.jl",
    "test_rho_pipi.jl",
    "test_Deltapi.jl",
    "test_ddstar.jl",
    "test_ddstar_pwave.jl",
    "test_dstardstar.jl",
    "test_dd_dstardstar.jl",
]

for filename in PHYSICS_TESTS
    module_name = Symbol("Physics_", replace(first(splitext(filename)), '-' => '_'))
    test_module = Module(module_name)
    Base.include(test_module, joinpath(@__DIR__, "physics", filename))
end
