using NPHFforFVE
using Test

@testset "free spectrum energy convention across backends" begin
    mass = 500.0
    channel = FockChannel(
        "scalar", [1], [:boson], [mass],
        [0//1], [0//1], [1.0], relativistic)

    # Filtering every channel pair forces the empty-interaction shortcut.
    potential = (args...) -> error("filtered potential must not be evaluated")
    channel_filter = (args...) -> false
    for (momentum, irrep) in ((D000, "A1+"), (D001, "A1"))
        project = Project(momentum, 0//1, [channel], [0])
        add_config!(project, 24, 0.1, [irrep], [1])
        momentum_squared = (2π * NPHFforFVE.ħc / (24 * 0.1))^2 *
                           sum(abs2, momentum)
        expected = sqrt(mass^2 + momentum_squared)
        for backend in (:complete_matrix, :projected_blocks, :factorized)
            for channel_decomp in (false, true)
                result = compute!(project, potential, nothing;
                                  backend=backend,
                                  channel_filter=channel_filter,
                                  channel_decomp=channel_decomp)
                @test result[1][irrep] ≈ [expected] atol=1e-8
            end
        end
    end
end
