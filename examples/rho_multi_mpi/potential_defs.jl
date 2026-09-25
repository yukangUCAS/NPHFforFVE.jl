# ============================================================
# Interaction Potential V — User-Defined Matrix Elements
# ============================================================
#
# Project Summary:
#   Total momentum: d = [0, 0, 0]
#   Total isospin: I = 1
#   ch 1: "rho" N=1 species=[1]
#   ch 2: "pipi" N=2 species=[2]
#   config 1: L=32, a=0.1, m_pi=208.0 MeV, Γ=T1-
#   config 2: L=48, a=0.1, m_pi=305.0 MeV, Γ=T1-
#
# Momentum convention: p = (2πħc/L_phys)n, with ħc = 197.327 MeV·fm.
# pA[i] and pB[i] are precomputed SVector{3,Float64} momenta in MeV.
# ============================================================

using NPHFforFVE

# ============ Params for Interaction (NOT the Params for Fit) ============
@params struct RhoPipiParams
    # the values are not actually used in the fitting.
    m_rho = 800.0
    m_pi = 208.0
    g = 1.8e-5
    h = 0.0
    cutoff = 900.0  # MeV
end

@inline function rho_pipi_polynomial(spin::Int, momentum)
    px, py, pz = momentum
    spin == 0 && return sqrt(3 / (4π)) * pz
    spin == 1 && return -sqrt(3 / (8π)) * (px + im * py)
    spin == -1 && return sqrt(3 / (8π)) * (px - im * py)
    return 0.0 + 0.0im
end

@inline function dipole_shape(momentum, cutoff::Real)
    cutoff_squared = cutoff^2
    return (cutoff_squared / (sum(abs2, momentum) + cutoff_squared))^2
end

@inline function pion_mass_factor(m_pi::Real)
    m_pi_physical = 140.0  # MeV
    return m_pi_physical^2 /
           (m_pi_physical^2 + (m_pi - m_pi_physical)^2)
end

# ============ Interaction Matrix Elements ============
#
# Signature:
#   my_V(nA, nB, sp, s, kapA, kapB, rA, rB, chA, chB, L_phys, params)
#
#   nA, nB       : bra and ket momenta; NTuple{N,Momentum}
#   sp, s        : bra and ket spin projections; NTuple{N,Rational{Int}}
#   kapA, kapB   : κ labels; String, or Tuple for multiple species
#   rA, rB       : physical subchannel indices at fixed κ
#   chA, chB     : channel indices listed below
#   L_phys       : spatial extent in fm
#   params       : RhoPipiParams instance
#
# Return the dim(κA) × dim(κB) interaction block; a scalar is allowed when both dimensions are 1.
#
# Hermiticity: my_V(nA,nB,...,chA,chB,...) = conj(my_V(nB,nA,...,chB,chA,...)).

# | Channel | Name | N | Species | κ at I = 1 |
# |---:|:---|---:|:---|:---|
# | 1 | rho | 1 | [1] | [1] |
# | 2 | pipi | 2 | [2] | [1,1] |

# Channel Details
#   ch 1: "rho" (1-body, relativistic)
#     sp.1: N=1, boson, m=params.m_rho MeV, j=1, I=1, η=-1.0
#     Isospin subchannels (Iₛ follows the species order above).
#     Intermediate isospins use left-associated coupling; — denotes direct coupling.
#
#     | κ | r | Subsystem isospins | dim(κ) |
#     |:---|---:|:---|---:|
#     | [1] | 1 | I₁ = 1 | 1 |
#
#     Charge-State Expansions at M = I = 1
#     For reference, the following equations give one transformation from the isospin basis to charge states.
#     |I=1, M=1; κ=[1], r=1⟩ = |1⟩
#   ch 2: "pipi" (2-body, relativistic)
#     sp.1: N=2, boson, m=params.m_pi MeV, j=0, I=1, η=1.0
#     Isospin subchannels (Iₛ follows the species order above).
#     Intermediate isospins use left-associated coupling; — denotes direct coupling.
#
#     | κ | r | Subsystem isospins | dim(κ) |
#     |:---|---:|:---|---:|
#     | [1,1] | 1 | I₁ = 1 | 1 |
#
#     Charge-State Expansions at M = I = 1
#     For reference, the following equations give one transformation from the isospin basis to charge states.
#     |I=1, M=1; κ=[1,1], r=1⟩ = −1/√2 |1,0⟩ + 1/√2 |0,1⟩

function my_V(nA, nB, sp, s, kapA, kapB, rA, rB, chA, chB, L_phys,
              params::RhoPipiParams)
    # ===== Momentum Conversion =====
    # p = (2πħc/L_phys)n, with ħc = 197.327 MeV·fm.
    pv = 2π * 197.327 / L_phys
    # pA[i] and pB[i] are SVector{3,Float64} momenta in MeV.
    pA = [pv .* Float64.(n) for n in nA]
    pB = [pv .* Float64.(n) for n in nB]
    # ── rho (1-body) diagonal ──
    if chA == 1 && chB == 1
        return 0.0 + 0.0im
    # ── rho ← pipi ──
    elseif chA == 1 && chB == 2
        relative_momentum = pB[1]
        momentum_scale = 500.0  # MeV; distinct from the dipole cutoff
        coupling = params.g + params.h * sum(abs2, relative_momentum) / momentum_scale^2
        vertex = coupling * rho_pipi_polynomial(Int(sp[1]), relative_momentum) *
                pion_mass_factor(params.m_pi) * dipole_shape(relative_momentum, params.cutoff)
        return ComplexF64(conj(vertex))

    # ── pipi ← rho  (Hermitian conjugate of 1←2) ──
    elseif chA == 2 && chB == 1
        # Hermitian conjugate: V(chB,chA) = conj(V(chA,chB))
        return conj(my_V(nB, nA, s, sp, kapB, kapA, rB, rA, chB, chA, L_phys, params))
    # ── pipi (2-body) diagonal ──
    elseif chA == 2 && chB == 2
        return 0.0 + 0.0im
    end
    error("unreachable: no matching channel pair for chA=$chA chB=$chB")
end

# ============ Optional Sparse Filters ============
# Return false only when the selected object is known to be exactly zero.
#
# Skip a complete Fock-channel pair.
function channel_filter(chA, chB, params)
    return chA != chB
end

# Skip one specific interaction matrix element.
function entry_filter(nA, nB, sp, s, kapA, kapB, rA, rB, chA, chB, L_phys, params)
    # nA[2] != nB[2] && return false  # spectator matching
    return true
end
