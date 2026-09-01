# NPHFforFVE.jl

NPHFforFVE.jl is a Julia package for computing finite-volume spectra of $N$-body systems ($N\le 4$). Users can define the system's Fock space and provide the corresponding interaction matrix elements. Based on these inputs, the program assembles the finite-volume Hamiltonian and projects it onto the specified irreducible representation of the finite-volume symmetry group, yielding the low-lying spectrum in each irreducible representation.


## Contents

- [NPHFforFVE.jl](#nphfforfvejl)
  - [Contents](#contents)
  - [Installation](#installation)
  - [Quick start](#quick-start)
    - [1. Define the system and generate an interaction template](#1-define-the-system-and-generate-an-interaction-template)
    - [2. Fill in the interaction](#2-fill-in-the-interaction)
    - [3. Compute and read the spectrum](#3-compute-and-read-the-spectrum)
  - [Core concepts](#core-concepts)
    - [FockChannel](#fockchannel)
    - [Project](#project)
    - [Config and add\_config!](#config-and-add_config)
  - [Isospin subchannels](#isospin-subchannels)
    - [Subchannel labels](#subchannel-labels)
    - [Inspecting subchannels and charge-state expansions](#inspecting-subchannels-and-charge-state-expansions)
    - [Excluding subchannels](#excluding-subchannels)
  - [Interaction interface](#interaction-interface)
    - [1. Parameters and signature](#1-parameters-and-signature)
    - [2. Helicity basis](#2-helicity-basis)
    - [3. Moving frame](#3-moving-frame)
  - [Results and compute!](#results-and-compute)
    - [Fock-channel decomposition](#fock-channel-decomposition)
    - [Output and saving](#output-and-saving)
  - [Computational backends and eigensolvers](#computational-backends-and-eigensolvers)
    - [1. `:complete_matrix`](#1-complete_matrix)
    - [2. `:projected_blocks`](#2-projected_blocks)
    - [3. `:factorized`](#3-factorized)
    - [4. Choosing a backend](#4-choosing-a-backend)
  - [Zero-entries filter (optional)](#zero-entries-filter-optional)
  - [Fitting spectra with NativeMinuit](#fitting-spectra-with-nativeminuit)
    - [1. Organizing fitting data](#1-organizing-fitting-data)
    - [2. A standard fit](#2-a-standard-fit)
    - [3. An accelerated fitting path for a common case](#3-an-accelerated-fitting-path-for-a-common-case)
  - [Complete example: three-coupled-channel Roper spectrum and fit](#complete-example-three-coupled-channel-roper-spectrum-and-fit)
    - [1. Computing the spectrum](#1-computing-the-spectrum)
    - [2. Parameter fitting](#2-parameter-fitting)
  - [Physics regression tests](#physics-regression-tests)
  - [License](#license)

## Installation

NPHFforFVE.jl requires Julia 1.10 or later. It can be installed directly from GitHub using Julia's package manager:

```julia
import Pkg
Pkg.add(url="https://github.com/yukangUCAS/NPHFforFVE.jl")
```

The package can then be loaded with:

```julia
using NPHFforFVE
```

To run the examples or tests in the source repository, or to participate in development, clone the repository and instantiate the project environment:

```bash
git clone https://github.com/yukangUCAS/NPHFforFVE.jl
cd NPHFforFVE.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Quick start

The following example considers two identical pion bosons in the rest frame with total isospin $I=2$. It computes low-lying levels in the `A1+` and `E+` irreducible representations of the finite-volume symmetry group $O_h$.

### 1. Define the system and generate an interaction template

Run the following in Julia:

```julia
using NPHFforFVE

# ππ Fock channel:
# 2 identical bosons, mass 140 MeV, spin 0, isospin 1, intrinsic parity -1
pipi = FockChannel(
    "pipi",
    [2],
    [:boson],
    [140.0],
    [0//1],
    [1//1],
    [-1.0],
    relativistic,
)

# Total momentum d=(0,0,0), total isospin I=2, ππ momentum cutoff Ncut=4
project = Project(D000, 2//1, [pipi], [4])

# L=48, a=0.1 fm; compute the four lowest A1+ levels and the two lowest E+ levels
config = add_config!(project, 48, 0.1, ["A1+", "E+"], [4, 2])

# Generate the interaction template from project
generate_potential_template(project, "pipi_potential.jl")
```

`FockChannel` describes the particle counts, statistical property, masses, spins, isospins, intrinsic parities, and kinetic-energy form of a Fock channel. `Project` specifies the total momentum, total isospin, participating Fock channels, and momentum cutoff for each channel.

Each call to `add_config!` defines a finite-volume setup and the irreducible representations to compute. Its return value is the index of that configuration in the result. Here the physical finite volume size is $L_{\mathrm{phys}}=48\times0.1\,\mathrm{fm}=4.8\,\mathrm{fm}$.

`generate_potential_template(project, "pipi_potential.jl")` creates initial template `pipi_potential.jl` in the current directory.

### 2. Fill in the interaction

Open the generated `pipi_potential.jl` and first modify the parameter structure as follows:

```julia
@params struct MyParams
    C0 = 0.5
    Lambda = 1000.0
end
```

Then locate the `TODO` for the diagonal `pipi` channel in the generated `my_V` function and replace that branch with:

```julia
if chA == 1 && chB == 1
    regulator = prod(
        1 / (1 + sum(abs2, p) / params.Lambda^2)^2
        for p in (pA..., pB...)
    )
    return ComplexF64(params.C0 * regulator)
end
```

The template automatically converts the dimensionless integer momenta `nA` and `nB` into momenta `pA` and `pB` in MeV, so they can be used directly to define the interaction.

The interaction function uses the following fixed call signature:

```julia
my_V(nA, nB, sp, s, kapA, kapB, rA, rB, chA, chB, L_phys, params)
```

Keeping the names generated by the template is recommended. Here, `nA`/`nB` and `sp`/`s` are the momenta and canonical-spin indices on the final/initial-state sides, respectively; `chA`/`chB` are Fock-channel indices. The function must return the matrix element between the corresponding isospin subchannels and satisfy the Hermiticity condition shown in the template. In this example the pion has spin zero and the isospin subchannel is one-dimensional, so the return value is a complex scalar.

### 3. Compute and read the spectrum

After saving `pipi_potential.jl`, run the following in the same Julia session in which `project` and `config` were defined:

```julia
include("pipi_potential.jl")

result = compute!(project, my_V, MyParams())
```

The result of `compute!` can be accessed as `result[config][irrep]`; each vector contains the finite-volume energy levels in ascending order, in MeV:

```julia
println("A1+ = ", round.(result[config]["A1+"], digits=4))
println("E+  = ", round.(result[config]["E+"], digits=4))
```

The output is:

```text
A1+ = [317.1743, 667.3997, 897.3895, 1049.0658]
E+  = [587.6023, 782.402]
```


## Core concepts

The basic workflow in NPHFforFVE.jl is:

```text
FockChannel
     ↓
Project
     ↓
add_config!
     ↓
generate_potential_template
     ↓
Interaction function + parameters
     ↓
compute!
     ↓
ProjectResult
```

If the system has nontrivial isospin subchannels, `generate_subchannel_report` can be used before generating the interaction template to inspect the subchannel information.

### FockChannel

The input format of `FockChannel` is shown below.

```julia
user_channel_1 = FockChannel(
    "channel_name",         # name: String, channel name
    [N1, N2, ...],           # species: Vector{Int}, number of particles of each species
    [:boson, :fermion, ...], # particle_types: Vector{Symbol}
    [m1, m2, ...],           # masses: fixed mass (MeV) or dynamic_mass(:field)
    [j1, j2, ...],     # spins: Vector{Rational{Int}}, single-particle spins
    [I1, I2, ...],     # isospins: Vector{Rational{Int}}, single-particle isospins
    [η1, η2, ...],           # etas: Vector{Float64}, must be +1.0 or -1.0
    relativistic,            # kinetic_type: relativistic or nonrelativistic
 )
```

`species` records the number of particles of each species. For example, `[1, 2]` denotes one particle of the first species and two particles of the second species. The sum of `species` gives the total number of particles in the channel. All vectors are ordered by particle species and have the same length.

Spins and isospins must be given as rational numbers. The currently supported single-particle spins and isospins are `0//1`, `1//2`, `1//1`, and `3//2`. A `FockChannel` can contain at most two particles with spin $\frac{3}{2}$ and at most two particles with isospin $\frac{3}{2}$.

Masses are usually given directly as positive real numbers in MeV. If a mass must vary with fit parameters, it should be written as `dynamic_mass(:field_name)`; every call to `compute!` reads its current value from `params.field_name`.

For example, the following defines a three-body $\rho\pi\pi$ Fock channel.

```julia
rho_pipi = FockChannel(
    "rho_pipi",              # name: Fock-channel name
    [1, 2],                   # species: 1 rho and 2 identical pions
    [:boson, :boson],         # particle_types: both species are bosons
    [770.0, 140.0],           # masses in MeV
    [1//1, 0//1],             # single-particle spins
    [1//1, 1//1],             # single-particle isospins
    [-1.0, -1.0],             # intrinsic parities
    relativistic,             # kinetic_type: relativistic kinetic energy
 )
```

A `Project` can contain one or more Fock channels.

### Project

`Project` stores the physical settings. Its constructor has the form:

```julia
Project(
    d,
    I,
    [user_channel_1, user_channel_2, ...],
    [Ncut_1, Ncut_2, ...];
    V_basis=:canonical,
)
```

Here, `d` is the three-dimensional integer total momentum, with allowed values `D000`, `D001`, `D011`, and `D111`; $I$ is the total isospin considered in the current project and must be given as a rational number, for example `2//1` or `1//2`; $\text{Ncut} := \left(p_\text{cut} L/2\pi\right)^2$ is the dimensionless momentum cutoff corresponding one-to-one with `channels`. (For a single-particle channel, the particle momentum equals the total momentum and its `Ncut` is ignored, so `0` can be supplied in the corresponding position.)

`V_basis` specifies the spin basis used for the later user-supplied interaction. The default `:canonical` denotes the canonical-spin basis, while `:helicity` denotes the helicity basis. When `generate_potential_template` is called, the program generates the corresponding interaction-function interface accordingly.

After fixing the total isospin, a many-particle system can still have multiple mutually orthogonal subchannels. Their physical meanings can be inspected with `generate_subchannel_report`; see the next section for details.

### Config and add_config!

Each call to `add_config!` add a specific configuration:

```julia
add_config!(project, L, a, [irrep_1, irrep_2, ...], [n_level_1, n_level_2, ...])
```

- `L`: dimensionless finite-volume size;
- `a`: lattice spacing in fm;
- `irrep`: name of an irreducible representation to compute;
- `n_level`: number of lowest energy levels to return for each irreducible representation.

The conventions for irreducible-representation strings used in this project are:

| Total momentum `d` | Little group (integer spin) | Available strings | Double cover for half-integer spin | Available strings |
|---|---|---|---|---|
| D000 = (0,0,0) | O_h | A1+, A2+, E+, T1+, T2+, A1-, A2-, E-, T1-, T2- | 2O_h | G1+, G2+, H+, G1-, G2-, H- |
| D001 = (0,0,1) | C4v | A1, A2, B1, B2, E | C4v^d | G1, G2 |
| D011 = (0,1,1) | C2v | A1, A2, B1, B2 | C2v^d | G |
| D111 = (1,1,1) | C3v | A1, A2, E | C3v^d | F1, F2, G |

A concrete example is:

```julia
config1 = add_config!(project, 48, 0.1, ["T1+", "E+"], [2, 3])
config2 = add_config!(project, 32, 0.07, ["E+"], [4])
```

After defining the interaction and computing the spectrum, these indices can be used to read the energy levels of the corresponding configurations:

```julia
result = compute!(project, my_V, params)

result[config1]["T1+"]
# alternatively, result[1]["T1+"]

result[config2]["E+"]
# alternatively, result[2]["E+"]
```

## Isospin subchannels

This section can be skipped if every `FockChannel` has only one subchannel at the specified total isospin.

Even after fixing the total isospin $I$, a many-particle isospin state need not be unique: states with different subsystem isospins or different isospin-coupling paths can be mutually orthogonal while having the same total isospin $I$. For example, in a $\rho\pi\pi$ channel with total isospin $1$, the $\pi\pi$ subsystem can have $I_{\pi\pi}=0,1,2$, each of which can couple with the rho to $I=1$. This project calls these states isospin subchannels.

### Subchannel labels

Each subchannel is labelled by `(κ,r)`:

- `κ` is an irreducible representation of the permutation group. When a `FockChannel` contains only one particle species, it is a string; when multiple particle species are present, it is a tuple of strings. For example, for $\rho\pi\pi\pi$, `kap = ("[1]", "[2,1]")` denotes `[1] ⊗ [2,1]`: the rho carries the `[1]` representation of $S_1$, while the three pions carry the `[2,1]` representation of $S_3$.
- `r` is the subchannel index at fixed `κ`. It further distinguishes isospins of subsystems, intermediate isospin-coupling paths, and further multiplicities.

### Inspecting subchannels and charge-state expansions

After creating a `Project`, a Markdown report containing all isospin-subchannel information can be generated with:

```julia
generate_subchannel_report(project, "subchannels.md")
```

The report contains:

- the physical information of the `Project` and its `FockChannel`s;
- all isospin subchannels of every `FockChannel`;
- the linear expansion coefficients between the isospin basis used by the project and the charge basis. These allow a user to convert a charge-basis interaction to the isospin basis used by the program:

$$
|I,I_z;\kappa,r,a\rangle
=\sum_{m_1,\ldots,m_N} C(m_1,\ldots,m_N)
|m_1,\ldots,m_N\rangle.
$$

### Excluding subchannels

If some subchannels are completely decoupled by symmetry and only contribute to trivial non-interacting energy levels, they can be excluded from the Hilbert space after creating the `Project`:

```julia
exclude_subchannel!(project, "rho_pipi", ("[1]", "[2]"), 1)
```

The entire subspace at a specified `κ` can also be excluded:

```julia
exclude_subchannel!(project, "rho_pipi", ("[1]", "[2]"))
```

`include_subchannel!` with the same form can remove an exclusion rule. The interaction template can then be generated:

```julia
generate_potential_template(project, "potential_defs.jl")
```

The final template and numerical basis contain the retained subchannels. `r` is not renumbered after exclusion. For example, if the original labels are `r=1,2` and `r=1` is excluded, the remaining subchannel is still labelled `r=2`.


## Interaction interface

**The current version accepts Hermitian interactions only.**

An interaction is jointly defined by a user-provided function and a parameter object. We recommend generating a template from an already configured `Project`:

```julia
generate_potential_template(project, "potential_defs.jl")
```

This function creates a new Julia file containing a parameter structure and a fixed interaction-function interface. The user then fills in their interaction matrix elements. After defining the interaction, do not run this command again for the same file, or the existing content will be overwritten.

Users only need to provide interaction matrix elements for each bra–ket subchannel branch listed in the template. In most cases, the function needs to return only a complex scalar; when multiple subchannels are present, it must return a matrix or vector.

### 1. Parameters and signature

The parameter structure in the template can be filled in according to the model. An interaction without parameters can also retain an empty structure:

```julia
@params struct MyParams
    C0 = 0.5
    Lambda = 1000.0
end
```

By default, the interaction uses the canonical-spin basis in spin space, with the fixed signature:

```julia
my_V(nA, nB, sp, s, kapA, kapB, rA, rB,
     chA, chB, L_phys, params)
```

The arguments have the following meanings:

| Argument | Meaning |
|---|---|
| `nA`, `nB` | Integer lattice-momentum tuples on the bra/ket sides, of type `NTuple{N, Momentum}`; the `i`th element corresponds to the `i`th particle |
| `sp`, `s` | Canonical-spin projections on the bra/ket sides, of type `NTuple{N, Rational{Int}}` |
| `kapA`, `kapB` | Permutation-group irreducible representations of the isospin subchannels; a string for one species and a tuple of strings for multiple species |
| `rA`, `rB` | Physical isospin-subchannel indices at fixed `kapA`/`kapB`; their physical meanings are given in the mapping table generated in the template |
| `chA`, `chB` | Fock-channel indices, starting from `1` |
| `L_phys` | Physical finite volume size in fm |
| `params` | Parameter object generated by `@params` |

`nA` and `nB` are dimensionless integer momenta, not momenta in MeV. In the rest frame, the physical momenta are

```julia
pv = 2π * 197.327 / L_phys
pA = [pv .* Float64.(n) for n in nA]
pB = [pv .* Float64.(n) for n in nB]
```

The generated template performs this conversion automatically.

The user only needs to replace the `TODO` in each branch with the appropriate matrix element. When either $\kappa_A$ or $\kappa_B$ is greater than 1, the interaction must return a $\dim(\kappa_A)\times\dim(\kappa_B)$ matrix. A size mismatch raises `DimensionMismatch`.

### 2. Helicity basis

When `V_basis=:helicity` is specified in `Project`, the generated template uses `my_V_hel` and replaces the canonical-spin arguments `sp` and `s` with helicity arguments `lamA` and `lamB`:

```julia
my_V_hel(nA, nB, lamA, lamB, kapA, kapB, rA, rB,
         chA, chB, L_phys, params)
```

This project adopts the following convention for single-particle helicity states:

$$
|\mathbf p,\lambda\rangle
=\sum_{\sigma=-j}^{j}
D^j_{\sigma\lambda}\!\left(R_{\mathrm{st}}(\mathbf p)\right)
|\mathbf p,\sigma\rangle,
\qquad
R_{\mathrm{st}}(\mathbf p)=R_z(\phi)R_y(\theta),
$$

where $(\theta,\phi)$ are the spherical coordinates of $\mathbf p$, and

$$
D^j_{\sigma\lambda}(\phi,\theta,0)
=e^{-i\sigma\phi}d^j_{\sigma\lambda}(\theta).
$$

A nonzero-spin particle at zero momentum has no definite helicity direction. Its corresponding component then automatically uses the canonical-spin projection `σ ∈ {j,j-1,...,-j}`. The generated template automatically identifies nonzero-spin particles at zero momentum on the bra and ket sides. For these particles, the corresponding entries of `lamA` and `lamB` are interpreted as canonical-spin projections. For all other particles, they are interpreted as helicities.

### 3. Moving frame

**Moving-frame calculations currently support two-body systems only.**

When `d != D000`, the template automatically calls `boost_to_cm` to obtain center-of-mass momenta `pA`/`pB`. Users only need to define the center-of-mass potential `V_cm` using the center-of-mass momenta. The template automatically generates the kinematic factors `facA`/`facB` and returns the interaction in the form

```julia
facA * V_cm * facB
```

## Results and compute!

The interaction interface defines the potential used to compute the spectrum (denoted by `my_V`) and the interaction parameters (denoted by `MyParams`). After calling

```julia
result = compute!(project, my_V, MyParams())
```

`result` is a `ProjectResult`. The spectrum for the `i`th configuration can be obtained as follows:

```julia
result[i]                 # dictionary of spectra for all irreducible representations in this configuration
result[i]["A1+"]           # energy levels in A1+, ordered from low to high energy
```

This uses the default backend `:complete_matrix`. For more complex large-matrix problems, `compute!` can also use matrix-free backends such as projected blocks or factorized. See the next section for details.

### Fock-channel decomposition

The weights of each eigenstate in each Fock channels can be accessed by setting `channel_decomp=true`:

```julia
result = compute!(project, my_V, MyParams();
                  backend=:factorized,
                  channel_decomp=true)
```

For example, the first `"A1+"` energy level in `config1` can be obtained with

```julia
weights = result.channel_decomp[config1]["A1+"][1]
```

and gives a result such as

```julia
Dict("rho" => 0.82, "pipi" => 0.18)
```

This means that, in the normalized projected eigenvector, about $82\%$ of the norm lies in the `"rho"` channel and $18\%$ in the `"pipi"` channel.

Accessing eigenvector information adds computational and storage overhead.

### Output and saving

To retain the complete `ProjectResult`, use Julia's standard-library `Serialization` module:

```julia
using Serialization

serialize("spectrum.jls", result)

# Read it again later
result_loaded = deserialize("spectrum.jls")
```

Typical access patterns are:

```julia
# All A1+ energy levels in configuration config1
energies = result_loaded[config1]["A1+"]

# Information for configuration config1
config_loaded = result_loaded.configs[config1]

# Parameter object used for the spectrum calculation
params_loaded = result_loaded.run_info.params

# Some Project settings
project_loaded = result_loaded.run_info.project

# If the original calculation used `channel_decomp=true`, the Fock-channel
# weights of an energy level can also be read:
weights = result_loaded.channel_decomp[config1]["A1+"][1]
```

This method saves configurations, spectra, run metadata, and any computed Fock-channel decompositions, and is suitable for continued analysis in Julia.

For convenient post-processing, the spectrum can also be written as a tab-separated text file:

```julia
write_spectrum(result, "spectrum.tsv")
```


## Computational backends and eigensolvers

The package provides three backends for solving the low-lying spectrum.

Select a backend explicitly with `backend`, for example:

```julia
result = compute!(project, my_V, MyParams();
                  backend=:complete_matrix)
```

Three built-in backends are available, as described below. The latter two use matrix-free eigensolvers.


### 1. `:complete_matrix`

This backend explicitly constructs the complete projected Hamiltonian and then uses LAPACK's Hermitian eigensolver.

```julia
result = compute!(project, my_V, MyParams();
                  backend=:complete_matrix)
```

Its computational path is direct. It is usually stable and efficient for small matrices, and is also suitable for validating results from the other backends. Its main limitation is that it must store a complete matrix. When matrix is large, memory consumption quickly becomes a bottleneck.

### 2. `:projected_blocks`

This backend precomputes and stores the projected interaction blocks but does not assemble them into one complete matrix. The Lanczos solver does not need the complete matrix. It only repeatedly computes the action of the Hamiltonian on a vector, $Hx$. Here, $Hx$ is obtained through blockwise matrix multiplication.

```julia
result = compute!(project, my_V, MyParams();
                  backend=:projected_blocks)
```

These projected blocks are partitioned according to the Fock-channel, isospin-subchannel, and momentum-orbit structure already present in the code. This backend is usually suitable for medium-scale systems or for interactions with many uncoupled channels and zero matrix blocks. Compared with `:complete_matrix`, it saves more memory; compared with `:factorized`, it stores more projected matrices but evaluates $Hx$ more directly. If all blocks are densely coupled, its memory advantage over the `:complete_matrix` is reduced.

### 3. `:factorized`

The program first constructs the interaction matrix $V_{\mathrm{unproj}}$ from the user-provided potential before spatial-symmetry projection. It then projects it into the subspace of irreducible representation $\Gamma$ using the projection matrix $Q$:

$$
V^\Gamma=Q^\dagger V_{\mathrm{unproj}}Q.
$$

This backend does not explicitly construct $V^\Gamma$. Each time $V^\Gamma x$ is needed, it performs

$$
x'=Qx,
\qquad
y'= V_{\mathrm{unproj}}x',
\qquad
y=Q^\dagger y',
$$

The lowest eigenvalues are likewise obtained with the Hermitian Lanczos method.

```julia
result = compute!(project, my_V, MyParams();
                  backend=:factorized)
```

This approach usually uses the least memory, especially for (A) the matrix is large but only a few lowest energy levels are needed. (B) $V_{\mathrm{unproj}}$ is sparse. Note that it still stores $V_{\mathrm{unproj}}$: if the unprojected interaction is itself a huge dense matrix, `:factorized` cannot eliminate this memory and matrix-multiplication cost.

### 4. Choosing a backend

| Situation | Recommended backend | Main reason |
|---|---|---|
| Small scale, or a benchmark result is needed | `:complete_matrix` | Direct computational path; suitable for validation |
| Medium scale, with clear blocks or zero couplings between channels and subchannels | `:projected_blocks` | Avoids assembling the complete $H^\Gamma$ while precomputing projected blocks |
| Large scale, only a few low-lying levels needed, or under memory pressure | `:factorized` | Does not store the projected interaction matrix |
| $V_{\mathrm{unproj}}$ is sparse | `:factorized` | Can use sparse matrix multiplication |
| $V_{\mathrm{unproj}}$ is huge and dense | No general optimal solution | All three backends must still bear interaction-construction or storage costs |

In practice, matrix dimension, block size, interaction sparsity, and the number of required energy levels all affect performance. For a new system, first compare the results of all three backends at relative smaller `Ncut`, then choose a strategy whose memory use is acceptable at the target scale. (Nonetheless, keep in mind that the performance of the backend also depends on `Ncut`.)

For huge unprojected interactions that are generally dense, there is currently no universally applicable optimized solver. The efficiency of such problems depends on the mathematical structure of the interaction, for example whether it has sparsity, separability, or affine parameter dependence. Algorithms must be designed and selected case by case for the specific model.

The default backend of `compute!` is `:complete_matrix`; the default backends of `prepare_spectrum` and `prepare_affine` are `:factorized`. In practice, we recommend specifying `backend` explicitly.

## Zero-entries filter (optional)

This is an optional performance optimization.

If the interaction contains many elements that are exactly zero, two filters can be used to skip the corresponding calculations in advance. As examples,

```julia
function channel_filter(chA, chB, params)
    # Return false when the complete Fock-channel pair is strictly uncoupled
    (chA, chB) in ((1, 3), (3, 1)) && return false # ch1 and ch3 are uncoupled
    return true
end

function entry_filter(nA, nB, sp, s, kapA, kapB, rA, rB,
                      chA, chB, L_phys, params)
    # Return false when this specific matrix element is strictly zero
    nA[2] != nB[2] && return false  # match a spectator momentum
    return true
end
```

Both filters must satisfy bra/ket exchange symmetry to satisfy the hermicity.

Pass the required filters explicitly (either one can be omitted independently):

```julia
result = compute!(project, my_V, MyParams();
                  channel_filter=channel_filter,
                  entry_filter=entry_filter)
```


## Fitting spectra with NativeMinuit

This package can be used together with `NativeMinuit.jl` to fit lattice spectra. First install NativeMinuit in the current Julia environment:

```julia
using Pkg
Pkg.add(url="https://github.com/fkguo/JuMinuit.jl")
```

NativeMinuit has additional Julia-version requirements, please refer to its official documentation. Then load both packages:

```julia
using NPHFforFVE
using NativeMinuit
```

### 1. Organizing fitting data

A fit commonly includes data from multiple finite-volume configurations and irreducible representations. Each data group is represented by a `SpectrumGroup`:

```julia
config1 = add_config!(project, 48, 0.10, ["A1+", "T1-"], [3, 2])
config2 = add_config!(project, 64, 0.08, ["A1+"], [2])

data = SpectrumDataset([
    SpectrumGroup(config1, "A1+", latdat_48_A1; covariance=C_48_A1), # 3 levels
    SpectrumGroup(config1, "T1-", latdat_48_T1; errors=sigma_T1), # 2 levels
    SpectrumGroup(config2, "A1+", latdat_64_A1; covariance=C_64_A1), # 2 levels
])
```

Here:

- `latdat_48_A1`, `latdat_48_T1`, and `latdat_64_A1` are the input spectra to be fitted;
- `errors` gives the standard deviations when energy levels are statistically independent, whereas `covariance` gives the covariance matrix of the levels within a Group. Use either as appropriate.

By default, the number of data points $N$ in each Group equals the corresponding number of requested levels in `add_config!`. Optionally, `levels` can specify the particular levels to be fitted. For example:

```julia
group = SpectrumGroup(config1, "A1+", [latdat1, latdat3];
                      levels=[1, 3], errors=[sigma1, sigma3])
```

This fits the first and third energy levels, excluding the second.

Different Groups are usually treated as statistically independent, so the total $\chi^2$ is the sum of their individual $\chi^2$ values. If correlations between `SpectrumGroup`s are known, provide a full global covariance matrix instead:

```julia
groups = [
    SpectrumGroup(config1, "A1+", latdat_48_A1),
    SpectrumGroup(config1, "T1-", latdat_48_T1),
    SpectrumGroup(config2, "A1+", latdat_64_A1),
]
data = SpectrumDataset(groups; covariance=C_global)
```

The rows and columns of `C_global` follow the order of `groups`, followed by the order of energy levels within each Group. When using a global covariance matrix, do not supply `errors` or `covariance` for individual Groups.

### 2. A standard fit

After organizing `data`, a standard fitting workflow is as follows.

First, precompute the quantities that need not be recomputed during the fit:

```julia
prepared = prepare_spectrum(project; backend=:factorized)
```

Then define the fitting problem:

```julia
# Use MyParams() as the starting point of the fit
problem = SpectrumFitProblem(prepared, my_V, MyParams(), data)
```

Here, `my_V` is the potential defined in the interaction template, and `MyParams()` is its corresponding parameter object. The values in `MyParams()` are also used as the fitting starting point.

`SpectrumFitProblem(..., data)` automatically constructs the objective function $\chi^2$ from the spectrum data and their errors or covariance matrix.

Then create and run NativeMinuit:

```julia
# Suppose that MyParams has parameters C0 and Lambda

m = minuit(problem;
    initial_steps=(C0=0.1, Lambda=5.0),  # Initial search step for each parameter
    limits=(Lambda=(0.0, Inf),),         # Allowed parameter range
    fixed=(Lambda=true,),                # Fixed parameters excluded from the fit
)

migrad!(m)
hesse!(m)

# Transfer the best-fit parameters back to MyParams and compute the corresponding spectrum

pbest = best_params(problem, m)
best_spectrum = compute!(prepared, my_V, pbest)
```

`migrad!(m)` searches for the parameters that minimize $\chi^2$. `hesse!(m)` computes parameter uncertainties and the covariance matrix near the resulting minimum. For asymmetric parameter errors, use `minos!`.

`m` is a native `NativeMinuit.Minuit` object. After fitting, it can be inspected directly:

```julia
m.valid       # Whether a valid minimum was found
m.fval        # chi^2 at the minimum
m.values      # Best-fit parameters in internal order
m.errors      # Parameter uncertainties from HESSE
m.covariance  # Parameter covariance matrix
```



### 3. An accelerated fitting path for a common case

If the interaction has the following strictly affine form in all fit parameters $\boldsymbol{\theta}$ ,

$$
V(\boldsymbol\theta)=V_0+\sum_i \theta_i V_i,
$$

replace `prepared` and `problem` in the standard workflow with:

```julia
prepared = prepare_affine(project, my_V, MyParams(); backend=:factorized)

problem = SpectrumFitProblem(prepared, MyParams(), data)
```

This mode precomputes and saves $V_0$ and every $V_i$ during preparation. During fitting, it only recombines these matrices using the current parameters. The potential must strictly satisfy the affine form above, and `channel_filter` and `entry_filter` must not depend on the fit parameters. If these conditions cannot be established, use the standard `prepare_spectrum` path.

If a `dynamic_mass` is among the fitting parameters, `prepare_affine` remains valid if and only if (1) the interaction does not depend on it and (2) the system is at rest.

Likewise, after fitting, calculate the spectrum at the best-fit parameters as follows:

```julia
pbest = best_params(problem, m)
best_spectrum = compute!(prepared, pbest)
```

## Complete example: three-coupled-channel Roper spectrum and fit

The [examples/roper](examples/roper) directory provides a runnable nontrivial example that demonstrates both spectrum calculation and parameter fitting. It considers the Roper system with total isospin $I=1/2$ in the rest-frame $G_1^+$ irrep, at two finite volumes $L=32,48$ and lattice spacing $a=0.091\,\mathrm{fm}$.

The three-coupled-channel model in this example follows J.-J. Wu, D. B. Leinweber, Z.-W. Liu, and A. W. Thomas, *Structure of the Roper Resonance from Lattice QCD Constraints*, *Physical Review D* **97**, 094509 (2018), [DOI: 10.1103/PhysRevD.97.094509](https://doi.org/10.1103/PhysRevD.97.094509) (also available as [arXiv:1703.10715](https://arxiv.org/abs/1703.10715)). The model is within two-body approximation.

The bare Roper mass is defined through `dynamic_mass(:m_Roper_bare)`. All particles in two-body channel are treated as stable in this example. The complete Fock-channel definitions are in [project.jl](examples/roper/project.jl); the potential and its parameter values are in [potential_defs.jl](examples/roper/potential_defs.jl).

### 1. Computing the spectrum

The example uses the following configuration:

```julia
project = Project(
    D000, 1//2,
    [Roper_bare, Npi, Deltapi, Nsigma],
    [0, 10, 10, 10],
)
add_config!(project, 32, 0.091, ["G1+"], [8])
add_config!(project, 48, 0.091, ["G1+"], [8])
```

After loading the project and potential definitions, the following computes the eight lowest $G_1^+$ levels at each volume:

```julia
params = RoperParams()
result = compute!(project, my_V, params;
    backend=:projected_blocks,
    channel_filter=channel_filter,
    entry_filter=entry_filter,
    channel_decomp=true,
)
write_spectrum(result, "spectrum.tsv")
```

Run the complete script directly:

```bash
julia --project=. examples/roper/spectra.jl
```

The results are written to [spectrum.tsv](examples/roper/spectrum.tsv).

### 2. Parameter fitting

For the fitting example, the script uses the spectrum computed in the previous step as the central values of pseudo-lattice data and assigns an independent uncertainty of $5\,\mathrm{MeV}$ to each level. The model's six regulator cutoffs and `g_Npi_Npi` are fixed at their reference values. The specific fixed parameters are listed in the `fixed` tuple below, and all remaining parameters are free. Each free parameter is initialized with a random $10\%\sim20\%$ relative perturbation from its reference value. If the fit succeeds, the best-fit parameters should return to the reference values used to generate the pseudo-data.

The remaining couplings enter the interaction affinely. In addition, the potential does not depend on the bare Roper mass, and this example is in the rest frame. Therefore, the affine path can accelerate the fit. The core code is:

```julia
fixed = (
    g_Npi_Npi=true,
    Lambda_R_Npi=true,
    Lambda_R_Deltapi=true,
    Lambda_R_Nsigma=true,
    Lambda_Npi=true,
    Lambda_Deltapi=true,
    Lambda_Nsigma=true,
)

prepared = prepare_affine(project, my_V, RoperParams();
    backend=:projected_blocks,
    channel_filter=channel_filter,
    entry_filter=entry_filter,
    fixed=fixed,
)
problem = SpectrumFitProblem(prepared, initial_params, data)
m = minuit(problem; initial_steps=initial_steps, fixed=fixed)

migrad!(m; iterate=1)

pbest = best_params(problem, m)
```

The complete code is in [fit_affine.jl](examples/roper/fit_affine.jl).

```bash
julia --project=. examples/roper/fit_affine.jl
```


## Physics regression tests

The public `test/physics/` directory covers representative systems with commonly used quantum numbers: integer and half-integer spins, identical bosons and fermions, two- and three-body systems, coupled channels, and rest and moving frames. Each test first constructs the Hamiltonian in the unprojected Fock space and obtains a reference spectrum, then projects it into every irreducible representation of the finite-volume symmetry group. It verifies that every eigenvalue in each projected irrep has a matching value in the unprojected reference spectrum, thereby testing the correctness of the package's projection method.

Run all physics regression tests with:

```bash
julia --project=. test/runtests.jl
```

The full suite contains several large bases and may take substantial time. Individual systems can also be run separately, for example:

```bash
julia --project=. test/physics/test_pipi.jl
```

## License

MIT License
