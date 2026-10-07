# D001 rho-pipi interaction. Include potential_defs.jl first for the shared
# RhoPipiParams type and center-of-mass vertex.
# config 1: L=32, a=0.1 fm, m_pi=208 MeV, irrep A1.
# Momentum boost and kinematic factors follow generate_potential_template.

function my_V_moving(nA, nB, sp, s, kapA, kapB, rA, rB, chA, chB,
                     L_phys, params::RhoPipiParams)
    chA == chB && return 0.0 + 0.0im
    pv = 2π * 197.327 / L_phys
    masses_by_ch = ([params.m_rho], [params.m_pi, params.m_pi])
    pA_mov = [pv .* Float64.(n) for n in nA]
    pB_mov = [pv .* Float64.(n) for n in nB]
    pA, facA = boost_to_cm(pA_mov, masses_by_ch[chA], D001, L_phys)
    pB, facB = boost_to_cm(pB_mov, masses_by_ch[chB], D001, L_phys)

    if chA == 1 && chB == 2
        vertex = rho_pipi_vertex(Int(sp[1]), pB[1], params)
        return facA * ComplexF64(conj(vertex)) * facB
    elseif chA == 2 && chB == 1
        vertex = rho_pipi_vertex(Int(s[1]), pA[1], params)
        return facA * ComplexF64(vertex) * facB
    end
    error("unreachable: no matching channel pair for chA=$chA chB=$chB")
end
