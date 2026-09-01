# ============================================================
# Roper interaction potential in the canonical-spin basis
# ============================================================
# Generated from examples/roper/project.jl.
#
# Project summary:
#   Total momentum d = (0,0,0), total isospin I = 1/2
#   ch 1: Roper_bare     ch 2: Npi
#   ch 3: Deltapi        ch 4: Nsigma
#   configuration: L = 32, a = 0.091 fm, irrep G1+
#
# The two-body channel order follows potential(Wu).jl: Npi, Deltapi, Nsigma.
# All channels have one one-dimensional isospin subchannel at I = 1/2, so
# kapA, kapB, rA, and rB carry no additional dependence in this model.
# ============================================================

using NPHFforFVE

const FPI = 92.4                 # MeV
const HBARC = 197.327            # MeV fm
const J_ROPER = 1//2
const M_ROPER = (1//2, -1//2)

# Indexed by the Project channel number. Channel 1 is the bare Roper.
const CHANNEL_L = (0, 1, 1, 0)
const CHANNEL_S = (0//1, 1//2, 3//2, 1//2)
const MESON_MASS = (0.0, 138.5, 138.5, 350.0) # MeV

# ============ Parameters ============
# Scenario II of the supplied reference parameter table. Couplings are kept
# dimensionless as in the source, while the cutoffs are stored in MeV.
@params struct RoperParams
    # Dynamic bare-state mass used by project.jl.
    m_Roper_bare = 1700.0

    # Bare Roper ↔ two-body vertices.
    g_R_Npi      = 0.954
    g_R_Deltapi  = -0.118
    g_R_Nsigma   = -2.892
    Lambda_R_Npi     = 630.2
    Lambda_R_Deltapi = 1431.8
    Lambda_R_Nsigma  = 1453.3

    # Symmetric direct two-body interactions.
    g_Npi_Npi         = 0.634
    g_Deltapi_Deltapi = -0.581
    g_Nsigma_Nsigma   = 10.000
    g_Npi_Deltapi     = -0.378
    g_Npi_Nsigma      = -1.738
    g_Deltapi_Nsigma  = 0.964
    Lambda_Npi      = 630.2
    Lambda_Deltapi  = 1431.8
    Lambda_Nsigma   = 1453.3
end

# ============ Radial and angular building blocks ============

@inline p2(p) = sum(abs2, p)
@inline meson_energy(p, ch::Int) = sqrt(p2(p) + MESON_MASS[ch]^2)
@inline regulator(p, Lambda::Float64) = exp(-p2(p) / Lambda^2)

@inline function channel_cutoff(ch::Int, p::RoperParams)
    ch == 2 && return p.Lambda_Npi
    ch == 3 && return p.Lambda_Deltapi
    ch == 4 && return p.Lambda_Nsigma
    throw(ArgumentError("channel $ch is not a two-body Roper channel"))
end

@inline function vertex_parameters(ch::Int, p::RoperParams)
    ch == 2 && return p.g_R_Npi, p.Lambda_R_Npi
    ch == 3 && return p.g_R_Deltapi, p.Lambda_R_Deltapi
    ch == 4 && return p.g_R_Nsigma, p.Lambda_R_Nsigma
    throw(ArgumentError("channel $ch is not a two-body Roper channel"))
end

@inline function direct_coupling(chA::Int, chB::Int, p::RoperParams)
    chA > chB && return direct_coupling(chB, chA, p)
    chA == 2 && chB == 2 && return p.g_Npi_Npi
    chA == 2 && chB == 3 && return p.g_Npi_Deltapi
    chA == 2 && chB == 4 && return p.g_Npi_Nsigma
    chA == 3 && chB == 3 && return p.g_Deltapi_Deltapi
    chA == 3 && chB == 4 && return p.g_Deltapi_Nsigma
    chA == 4 && chB == 4 && return p.g_Nsigma_Nsigma
    throw(ArgumentError("invalid two-body channel pair ($chA, $chB)"))
end

# p^L Y_Lm(p-hat) for L = 0, 1. This solid-harmonic form is regular at p = 0:
# the P-wave components vanish there without forming p-hat explicitly.
@inline function solid_harmonic(L::Int, m::Rational{Int}, p)
    L == 0 && m == 0 && return ComplexF64(sqrt(1 / (4π)))
    L == 1 && m == -1 && return sqrt(3 / (8π)) * (p[1] - im * p[2])
    L == 1 && m == 0  && return ComplexF64(sqrt(3 / (4π)) * p[3])
    L == 1 && m == 1  && return -sqrt(3 / (8π)) * (p[1] + im * p[2])
    return 0.0 + 0.0im
end

# p^L Y_Lm(p-hat) ⟨Lm,Sσ|1/2,M⟩ for the JMLS → canonical-spin conversion.
@inline function jls_coefficient(p, ch::Int,
                                 sigma::Rational{Int}, M::Rational{Int})
    L, S = CHANNEL_L[ch], CHANNEL_S[ch]
    m = M - sigma
    abs(m) <= L || return 0.0 + 0.0im
    cg = NPHFforFVE._su2_clebsch_gordan(L // 1, m, S, sigma, J_ROPER, M)
    return solid_harmonic(L, m, p) * cg
end

@inline function bare_to_two_body(pvec, ch::Int,
                                  sigma::Rational{Int}, M::Rational{Int},
                                  p::RoperParams)
    g, Lambda = vertex_parameters(ch, p)
    L = CHANNEL_L[ch]
    radial = g / (2π * FPI^L) * regulator(pvec, Lambda) /
             sqrt(meson_energy(pvec, ch))
    return radial * jls_coefficient(pvec, ch, sigma, M)
end

@inline function two_body_to_two_body(pA, pB, chA::Int, chB::Int,
                                      sigmaA::Rational{Int}, sigmaB::Rational{Int},
                                      p::RoperParams)
    g = direct_coupling(chA, chB, p)
    LA, LB = CHANNEL_L[chA], CHANNEL_L[chB]
    radial = g / (4π^2 * FPI^(LA + LB)) *
             regulator(pA, channel_cutoff(chA, p)) *
             regulator(pB, channel_cutoff(chB, p)) /
             (meson_energy(pA, chA) * meson_energy(pB, chB))

    angular = 0.0 + 0.0im
    for M in M_ROPER
        angular += jls_coefficient(pA, chA, sigmaA, M) *
                   conj(jls_coefficient(pB, chB, sigmaB, M))
    end
    return radial * angular
end

@inline function relative_momentum(n, L_phys::Real)
    # At d = 0 particle 2 is the pion or sigma, hence its momentum is the
    # relative momentum used by the JMLS interaction.
    return (2π * HBARC / L_phys) * n[2]
end

# ============ Interaction matrix element ============
#
# my_V(nA, nB, sp, s, kapA, kapB, rA, rB, chA, chB, L_phys, params)
#
# nA/nB and sp/s are bra/ket momenta and canonical spin projections. For a
# two-body channel, spin projection 1 belongs to N or Delta and projection 2
# is the spin-zero pion or sigma.
function my_V(nA, nB, sp, s, kapA, kapB, rA, rB,
              chA::Int, chB::Int, L_phys::Real, p::RoperParams)
    (1 <= chA <= 4 && 1 <= chB <= 4) ||
        throw(ArgumentError("Roper channel indices must lie in 1:4"))

    chA == 1 && chB == 1 && return 0.0 + 0.0im

    if chA == 1
        # ⟨R,M|V|c,p,σ⟩ = conj(⟨c,p,σ|V|R,M⟩).
        pB = relative_momentum(nB, L_phys)
        return conj(bare_to_two_body(pB, chB, s[1], sp[1], p))
    elseif chB == 1
        pA = relative_momentum(nA, L_phys)
        return bare_to_two_body(pA, chA, sp[1], s[1], p)
    end

    pA = relative_momentum(nA, L_phys)
    pB = relative_momentum(nB, L_phys)
    return two_body_to_two_body(pA, pB, chA, chB, sp[1], s[1], p)
end

# ============ Optional sparse filters ============
# The bare one-body channel has no direct diagonal interaction.
channel_filter(chA, chB, params) = !(chA == 1 && chB == 1)

# No other matrix element is known to vanish independently of the parameters.
entry_filter(nA, nB, sp, s, kapA, kapB, rA, rB, chA, chB, L_phys, params) = true
