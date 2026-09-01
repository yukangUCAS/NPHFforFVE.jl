using NPHFforFVE

# Particle masses in MeV.
m_N = 938.0
m_pi = 138.5
m_sigma = 350.0
m_Delta = 1232.0
# The bare Roper mass is a fit parameter, read as params.m_Roper_bare at
# every spectrum evaluation.
m_Roper_bare = dynamic_mass(:m_Roper_bare)

# Bare Roper: one stable fermion with I(J^P) = 1/2(1/2+).
Roper_bare = FockChannel(
    "Roper_bare",
    [1],
    [:fermion],
    [m_Roper_bare],
    [1//2],
    [1//2],
    [1.0],
    relativistic,
)

# N-pi channel. Particle order: (N, pi).
Npi = FockChannel(
    "Npi",
    [1, 1],
    [:fermion, :boson],
    [m_N, m_pi],
    [1//2, 0//1],
    [1//2, 1//1],
    [1.0, -1.0],
    relativistic,
)

# N-sigma channel. Particle order: (N, sigma).
Nsigma = FockChannel(
    "Nsigma",
    [1, 1],
    [:fermion, :boson],
    [m_N, m_sigma],
    [1//2, 0//1],
    [1//2, 0//1],
    [1.0, 1.0],
    relativistic,
)

# Delta-pi channel. Particle order: (Delta, pi).
Deltapi = FockChannel(
    "Deltapi",
    [1, 1],
    [:fermion, :boson],
    [m_Delta, m_pi],
    [3//2, 0//1],
    [3//2, 1//1],
    [1.0, -1.0],
    relativistic,
)

# Roper sector: rest frame, total isospin I=1/2.
# The single-particle channel has no independent momentum cutoff, so its entry is 0.
project = Project(
    D000,
    1//2,
    # Channel order follows potential(Wu).jl: R_bare, Npi, Deltapi, Nsigma.
    [Roper_bare, Npi, Deltapi, Nsigma],
    [0, 10, 10, 10],
)

# Finite-volume spectrum: a = 0.091 fm, L = 32, rest-frame G1+ sector.
# Request the lowest eight levels in this irrep.
config = add_config!(project, 32, 0.091, ["G1+"], [8])
config2 = add_config!(project, 48, 0.091, ["G1+"], [8])
