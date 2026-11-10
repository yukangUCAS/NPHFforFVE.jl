using Test

module GeneratedPotentialTemplateTests
using Test, NPHFforFVE

@testset "Generated potential Hermitian conjugate" begin
    channels = [FockChannel("three$i", [3], [:boson], [140.0], [0], [1],
                            [1.0], relativistic) for i in 1:2]
    momenta = (Momentum(0, 0, 0), Momentum(1, 0, 0), Momentum(-1, 0, 0))
    spins = (0//1, 0//1, 0//1)
    cases = [
        (1, 1, 2 + 3im),
        (1, 1, reshape(ComplexF64[2 + 3im], 1, 1)),
        (2, 2, ComplexF64[1 im; 2 + im 3]),
        (2, 1, reshape(ComplexF64[1, 2im], 2, 1)),
        (1, 2, reshape(ComplexF64[1, 2im], 1, 2)),
        (2, 1, ComplexF64[1, 2im]),
        (1, 2, ComplexF64[1, 2im]),
        (1, 1, ComplexF64[2 + 3im]),
        (2, 2, 0.0),
        (2, 2, adjoint(ComplexF64[1 im; 3 2])),
    ]
    for mode in (:canonical, :helicity)
        project = Project(D000, 1//1, channels, [1, 1]; V_basis=mode)
        mktempdir() do dir
            path = generate_potential_template(project, joinpath(dir, "potential.jl"))
            # Fill only the user TODOs; execute the generated reverse branch unchanged.
            source = replace(read(path, String),
                r"error\(\"TODO: fill matrix block[^\n]*\)" => "return params")
            generated = Module(gensym(:PotentialTemplate))
            Base.include_string(generated, source, path)
            callback = getfield(generated, mode == :canonical ? :my_V : :my_V_hel)
            for (dimA, dimB, value) in cases
                kapA = dimA == 1 ? "[3]" : "[2,1]"
                kapB = dimB == 1 ? "[3]" : "[2,1]"
                forward = NPHFforFVE._V_adapter(callback, kapA, kapB, 1, 1,
                    dimA, dimB, 1, 2, 4.8, value)
                reverse = NPHFforFVE._V_adapter(callback, kapB, kapA, 1, 1,
                    dimB, dimA, 2, 1, 4.8, value)
                A = Base.invokelatest(forward, momenta, spins, momenta, spins)
                B = Base.invokelatest(reverse, momenta, spins, momenta, spins)
                @test size(B) == (B isa Number ? () : (dimB, dimA))
                @test B ≈ adjoint(A)
            end
        end
    end
end
end

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

Base.include(Module(:MultiMpiFitTest), joinpath(@__DIR__, "test_multi_mpi_fit.jl"))
Base.include(Module(:MovingFreeBackendTest), joinpath(@__DIR__, "test_moving_free_backends.jl"))
Base.include(Module(:JointFitTest), joinpath(@__DIR__, "test_joint_fit.jl"))
