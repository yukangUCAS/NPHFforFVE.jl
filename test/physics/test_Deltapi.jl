using NPHFforFVE
using LinearAlgebra
using Test

const M_Dpi = NPHFforFVE.Momentum
const SG_Dpi = NPHFforFVE.SymmetryGroup
const m_Delta_Dpi = 1232.0
const m_pi_Dpi = 138.5
const L_Dpi = 24
const a_Dpi = 0.1
const Ncut_Dpi = 15
const hbarc_Dpi = 197.327
const Lambda_Dpi = 900.0
const species_Dpi = [1, 1]
const particle_types_Dpi = [:fermion, :boson]
const spins_Dpi = Rational{Int}[3//2, 0//1]
const spins_float_Dpi = Float64[1.5, 0.0]
const etas_Dpi = Float64[1.0, -1.0]
const kappa_Dpi = ("[1]", "[1]")
const irreps_Dpi = NPHFforFVE.OH2_IRREP_NAMES
const momentum_prefactor_Dpi = (2pi * hbarc_Dpi / (L_Dpi * a_Dpi))^2
const fv_factor_Dpi = (2pi * hbarc_Dpi / (L_Dpi * a_Dpi))^3
const Lambda_n2_Dpi = Lambda_Dpi^2 / momentum_prefactor_Dpi
const identity4_Dpi = Matrix{ComplexF64}(I, 4, 4)

@params struct DeltaPiTestParams
    C0 = 8.0e-6
    Cp = 1.5e-6
    Cs = -2.0e-6
end

regulator_Dpi(q) = 1.0 / (1.0 + sum(abs2, q) / Lambda_n2_Dpi)^2

function canonical_block_Dpi(nA, nB, p::DeltaPiTestParams)
    qA = Float64.(nA[1])
    qB = Float64.(nB[1])
    SdotA = qA[1] * SG_Dpi._JX_THREEHALF +
            qA[2] * SG_Dpi._JY_THREEHALF +
            qA[3] * SG_Dpi._JZ_THREEHALF
    SdotB = qB[1] * SG_Dpi._JX_THREEHALF +
            qB[2] * SG_Dpi._JY_THREEHALF +
            qB[3] * SG_Dpi._JZ_THREEHALF
    form_factor = regulator_Dpi(qA) * regulator_Dpi(qB)
    return form_factor .* (
        (p.C0 + p.Cp * dot(qA, qB)) .* identity4_Dpi +
        p.Cs .* (SdotA * SdotB))
end

spin_index_Dpi(sigma) = Int(3//2 - Rational{Int}(sigma)) + 1

function V_Deltapi(nA, nB, sp, s, kapA, kapB, rA, rB,
                   chA, chB, L_phys, p::DeltaPiTestParams)
    sp[2] == 0 && s[2] == 0 || return 0.0 + 0.0im
    block = canonical_block_Dpi(nA, nB, p)
    return block[spin_index_Dpi(sp[1]), spin_index_Dpi(s[1])]
end

function V_Deltapi_hel(nA, nB, lamA, lamB, kapA, kapB, rA, rB,
                       chA, chB, L_phys, p::DeltaPiTestParams)
    cA = get_rotation_vector(nA, lamA, spins_Dpi)
    cB = get_rotation_vector(nB, lamB, spins_Dpi)
    return dot(cA, canonical_block_Dpi(nA, nB, p) * cB)
end

function _distinct_Dpi(values; atol=1e-8)
    result = Float64[]
    for value in sort(Float64.(values))
        (isempty(result) || value - result[end] > atol) && push!(result, value)
    end
    return result
end

function _reference_Dpi(p::DeltaPiTestParams)
    representatives = find_representatives(
        2; Ncut=Ncut_Dpi, d=D000, species=species_Dpi,
        particle_types=particle_types_Dpi)
    all_states = Any[]
    seen_states = Dict{Any,Int}()

    for representative in representatives
        helicities = try
            helicity_representatives(
                representative; species=species_Dpi,
                particle_types=particle_types_Dpi, spins=spins_Dpi, d=D000)
        catch
            [(0.0, 0.0)]
        end
        for helicity in helicities, irrep in irreps_Dpi
            projected = subspace_projection(
                representative, Tuple(Float64.(helicity)), kappa_Dpi, irrep;
                d_total=D000, species=species_Dpi,
                particle_types=particle_types_Dpi,
                spins=spins_float_Dpi, etas=etas_Dpi)
            isempty(projected.Z) && continue
            for state in projected.subspace_states
                if !haskey(seen_states, state)
                    push!(all_states, state)
                    seen_states[state] = length(all_states)
                end
            end
        end
    end

    kinetic = zeros(ComplexF64, length(all_states), length(all_states))
    masses = (m_Delta_Dpi, m_pi_Dpi)
    for (index, (momenta, _)) in enumerate(all_states)
        kinetic[index, index] = sum(
            sqrt(mass^2 + momentum_prefactor_Dpi * Float64(sum(abs2, momentum)))
            for (mass, momentum) in zip(masses, momenta))
    end
    canonical_adapter(nA, sp, nB, s, args...) =
        canonical_block_Dpi(nA, nB, p)[spin_index_Dpi(sp[1]), spin_index_Dpi(s[1])]
    interaction = build_V_hel(
        all_states, all_states, spins_Dpi, spins_Dpi, canonical_adapter)
    hamiltonian = Hermitian(kinetic + fv_factor_Dpi .* interaction)
    return (states=all_states,
            eigenvalues=sort(real.(eigvals(hamiltonian))))
end

function _Dpi_channel()
    return FockChannel(
        "Deltapi", [1, 1], [:fermion, :boson],
        [m_Delta_Dpi, m_pi_Dpi], [3//2, 0//1], [3//2, 1//1],
        [1.0, -1.0], relativistic)
end

@testset "Delta-pi nontrivial Hermitian interaction" begin
    p = DeltaPiTestParams()
    sample_momenta = [
        (M_Dpi(1, 0, 0), M_Dpi(-1, 0, 0)),
        (M_Dpi(0, 1, 0), M_Dpi(0, -1, 0)),
        (M_Dpi(1, 1, 0), M_Dpi(-1, -1, 0))]
    spin_values = (3//2, 1//2, -1//2, -3//2)
    has_spin_mixing = false
    has_complex_entry = false
    for nA in sample_momenta, nB in sample_momenta
        block_AB = canonical_block_Dpi(nA, nB, p)
        block_BA = canonical_block_Dpi(nB, nA, p)
        @test block_AB ≈ adjoint(block_BA) atol=1e-14
        has_spin_mixing |= any(abs(block_AB[i, j]) > 1e-14
                               for i in 1:4, j in 1:4 if i != j)
        has_complex_entry |= any(abs(imag(value)) > 1e-14 for value in block_AB)
    end
    @test has_spin_mixing
    @test has_complex_entry

    for nA in sample_momenta, nB in sample_momenta,
        sp in spin_values, s in spin_values
        canonical_value = V_Deltapi(
            nA, nB, (sp, 0//1), (s, 0//1), kappa_Dpi, kappa_Dpi,
            1, 1, 1, 1, L_Dpi * a_Dpi, p)
        reverse_value = V_Deltapi(
            nB, nA, (s, 0//1), (sp, 0//1), kappa_Dpi, kappa_Dpi,
            1, 1, 1, 1, L_Dpi * a_Dpi, p)
        @test canonical_value ≈ conj(reverse_value) atol=1e-14
    end
end

@testset "Delta-pi projected spectrum matches the unprojected spectrum" begin
    p = DeltaPiTestParams()
    reference = _reference_Dpi(p)
    channel = _Dpi_channel()
    system = FockSystem(
        D000, Ncut_Dpi, [channel], L_Dpi, a_Dpi, 1//2, irreps_Dpi)

    matched_dimensions = 0
    for irrep in irreps_Dpi
        projected_hamiltonian = build_hamiltonian_block(
            system, irrep, V_Deltapi, p)
        isempty(projected_hamiltonian) && continue
        projected_values = sort(real.(eigvals(Hermitian(projected_hamiltonian))))
        @test all(value -> minimum(abs.(reference.eigenvalues .- value)) < 2e-7,
                  projected_values)
        matched_dimensions += length(projected_values) *
                              size(irrep_matrix(irrep, 1; group=:Oh2), 1)
    end
    @test matched_dimensions == length(reference.eigenvalues)
    @test length(_distinct_Dpi(reference.eigenvalues)) > 4
end

@testset "Delta-pi backends, helicity input, and prepared paths" begin
    p = DeltaPiTestParams()
    channel = _Dpi_channel()
    project = Project(D000, 1//2, [channel], [Ncut_Dpi])
    config = add_config!(project, L_Dpi, a_Dpi, ["G1+", "H+"], [3, 3])

    results = Dict(backend => compute!(
        project, V_Deltapi, p; backend=backend, validate_hermitian=true)
        for backend in (:complete_matrix, :projected_blocks, :factorized))
    for backend in (:projected_blocks, :factorized), irrep in ("G1+", "H+")
        @test results[backend][config][irrep] ≈
              results[:complete_matrix][config][irrep] atol=2e-7 rtol=1e-9
    end

    helicity_project = Project(
        D000, 1//2, [channel], [Ncut_Dpi]; V_basis=:helicity)
    helicity_config = add_config!(
        helicity_project, L_Dpi, a_Dpi, ["G1+", "H+"], [3, 3])
    helicity_result = compute!(
        helicity_project, V_Deltapi_hel, p;
        backend=:complete_matrix, validate_hermitian=true)
    for irrep in ("G1+", "H+")
        @test helicity_result[helicity_config][irrep] ≈
              results[:complete_matrix][config][irrep] atol=2e-7 rtol=1e-9
    end

    prepared = prepare_spectrum(project; backend=:factorized)
    prepared_result = compute!(prepared, V_Deltapi, p; channel_decomp=true)
    for irrep in ("G1+", "H+")
        @test prepared_result[config][irrep] ≈
              results[:factorized][config][irrep] atol=2e-7 rtol=1e-9
        for weights in prepared_result.channel_decomp[config][irrep]
            @test weights["Deltapi"] ≈ 1.0 atol=2e-10
        end
    end

    affine = prepare_affine(project, V_Deltapi, DeltaPiTestParams();
                            backend=:factorized)
    shifted = DeltaPiTestParams(C0=7.2e-6, Cp=-0.8e-6, Cs=2.4e-6)
    affine_result = compute!(affine, shifted)
    direct_shifted = compute!(
        project, V_Deltapi, shifted; backend=:factorized)
    for irrep in ("G1+", "H+")
        @test isapprox(affine_result[config][irrep],
                       direct_shifted[config][irrep]; atol=2e-7, rtol=1e-9)
    end
end
