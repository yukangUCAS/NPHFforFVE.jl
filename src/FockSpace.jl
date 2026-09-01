module FockSpace

using ..NPHFforFVE: Momentum, D000, _to_momentum, isospin_decomposition, multi_isospin_decomposition, get_SN_irrep_dim
using ..NPHFforFVE: cache_channel_reps!
using ..NPHFforFVE: group_for_momentum
using ..NPHFforFVE: OH_IRREP_NAMES, OH2_IRREP_NAMES, LG_IRREP_NAMES

export FockChannel, FockSystem, setup_fock_system
export get_N, get_num_species, get_total_N
export get_isospin_subchannels
export SubchannelExclusion
export KineticType, relativistic, nonrelativistic
export DynamicMass, dynamic_mass, has_dynamic_mass
export resolve_mass, resolve_masses, resolve_particle_masses

# ============ Kinetic-energy dispersion ============

"""
    KineticType

Enumeration specifying the kinetic-energy dispersion.

- `relativistic`: E(p) = √(m² + p²)
- `nonrelativistic`: E(p) = m + p²/(2m)
"""
@enum KineticType begin
    relativistic
    nonrelativistic
end

# ============ FockChannel ============

"""A particle mass supplied by `params.<name>` at evaluation time."""
struct DynamicMass
    name::Symbol

    function DynamicMass(name::Symbol)
        isempty(String(name)) && throw(ArgumentError(
            "dynamic mass parameter name must not be empty"))
        new(name)
    end
end

"""
    dynamic_mass(name::Symbol) -> DynamicMass

Mark a `FockChannel` mass as dynamic. Every spectrum evaluation resolves it
from the field `params.<name>`.
"""
dynamic_mass(name::Symbol) = DynamicMass(name)

function Base.show(io::IO, mass::DynamicMass)
    print(io, "dynamic_mass(", repr(mass.name), ")")
end

const MassSpec = Union{Float64,DynamicMass}

function _normalize_mass(mass, species_index::Int)
    if mass isa DynamicMass
        return mass
    elseif mass isa Real && !(mass isa Bool)
        value = Float64(mass)
        isfinite(value) && value > 0 || throw(ArgumentError(
            "Species $species_index: mass must be finite and > 0 MeV, got $mass"))
        return value
    end
    throw(ArgumentError(
        "Species $species_index: mass must be a real number or DynamicMass, got $(typeof(mass))"))
end

"""
    FockChannel(name, species, particle_types, masses, spins, isospins)

A single Fock channel. The total isospin `I` is global and stored in `FockSystem`.

# Arguments
- `name::String`: channel name, for example `\"ππN\"`
- `species::Vector{Int}`: particle count for each species
- `particle_types::Vector{Symbol}`: particle type for each species (`:boson` or `:fermion`)
- `masses`: mass of each species in MeV; use a positive real number for a fixed mass or `dynamic_mass(:parameter_name)` for a dynamic mass
- `spins::Vector{Rational{Int}}`: spin `j` of each species
- `isospins::Vector{Rational{Int}}`: isospin `j` of each species
"""
struct FockChannel
    name::String
    N::Int
    species::Vector{Int}
    particle_types::Vector{Symbol}
    masses::Union{Vector{Float64},Vector{MassSpec}}
    spins::Vector{Rational{Int}}
    isospins::Vector{Rational{Int}}
    etas::Vector{Float64}
    kinetic_type::KineticType

    function FockChannel(name::AbstractString, species::Vector{Int},
                         particle_types::Vector{Symbol}, masses,
                         spins, isospins, etas, kinetic_type::KineticType)
        n = length(species)
        name     = String(name)
        normalized_masses = MassSpec[
            _normalize_mass(m, i) for (i, m) in enumerate(masses)]
        masses = all(m -> m isa Float64, normalized_masses) ?
            Float64[normalized_masses...] : normalized_masses
        spins    = Rational{Int}.(spins)
        isospins = Rational{Int}.(isospins)
        etas     = Float64.(etas)
        all(x -> x > 0, species) || throw(ArgumentError("each species must contain at least one particle"))
        length(particle_types) == n || throw(ArgumentError("particle_types must have the same length as species"))
        length(masses) == n        || throw(ArgumentError("masses must have the same length as species"))
        length(spins) == n         || throw(ArgumentError("spins must have the same length as species"))
        length(isospins) == n      || throw(ArgumentError("isospins must have the same length as species"))
        length(etas) == n          || throw(ArgumentError("etas must have the same length as species"))
        for (i, pt) in enumerate(particle_types)
            pt in (:boson, :fermion) || throw(ArgumentError(
                "Species $i: particle_type must be :boson or :fermion, got :$pt"))
        end
        for (i, eta) in enumerate(etas)
            eta ∈ (-1.0, 1.0) || throw(ArgumentError(
                "Species $i: intrinsic parity eta must be +1 or -1, got $eta"))
        end
        _validate_spins(spins)
        _validate_isospins(isospins)
        N = sum(species)
        new(name, N, species, particle_types, masses, spins, isospins, etas, kinetic_type)
    end
end

has_dynamic_mass(ch::FockChannel) = any(m -> m isa DynamicMass, ch.masses)

resolve_mass(mass::Float64, params=nothing) = mass

function resolve_mass(mass::DynamicMass, params)
    params === nothing && throw(ArgumentError(
        "dynamic mass $(repr(mass.name)) requires a parameter object"))
    hasproperty(params, mass.name) || throw(ArgumentError(
        "dynamic mass $(repr(mass.name)) requires parameter field `$(mass.name)`, " *
        "but $(typeof(params)) does not contain it"))
    raw = getproperty(params, mass.name)
    raw isa Real && !(raw isa Bool) || throw(ArgumentError(
        "dynamic mass parameter `$(mass.name)` must be a real number, got $(typeof(raw))"))
    value = Float64(raw)
    isfinite(value) && value > 0 || throw(ArgumentError(
        "dynamic mass parameter `$(mass.name)` must be finite and > 0 MeV, got $raw"))
    return value
end

function resolve_masses(ch::FockChannel, params=nothing)
    result = Vector{Float64}(undef, length(ch.masses))
    for i in eachindex(ch.masses)
        try
            result[i] = resolve_mass(ch.masses[i], params)
        catch err
            err isa ArgumentError || rethrow()
            throw(ArgumentError(
                "Fock channel $(repr(ch.name)), species $i: $(sprint(showerror, err))"))
        end
    end
    return result
end

function resolve_particle_masses(ch::FockChannel, params=nothing)
    species_masses = resolve_masses(ch, params)
    result = Float64[]
    sizehint!(result, ch.N)
    for (count, mass) in zip(ch.species, species_masses)
        append!(result, fill(mass, count))
    end
    return result
end

function _validate_quantum_numbers(label::String,
                                   vals::Vector{Rational{Int}},
                                   allowed)
    for (i, v) in enumerate(vals)
        v in allowed || throw(ArgumentError(
            "Species $i: $label = $v is not currently supported; only " *
            join(allowed, ", ")))
    end
end

_validate_spins(vals::Vector{Rational{Int}}) =
    _validate_quantum_numbers("spin", vals, (0, 1//2, 1, 3//2))

_validate_isospins(vals::Vector{Rational{Int}}) =
    _validate_quantum_numbers("isospin", vals, (0, 1//2, 1, 3//2))

get_N(ch::FockChannel) = ch.N
get_num_species(ch::FockChannel) = length(ch.species)

# ============ Isospin subchannels ============

"""
    IsospinSubChannel

An isospin subchannel, identified by `(κ, r)`.

- `κ`: S_N irrep label. For one species: a `String` (for example `"[2,1]"`);
       for multiple species: `NTuple{K,String}` (for example `("[2]", "[1]")` for S₂×S₁).
- `r::Int`: unique physical-subchannel index for fixed κ (1 ≤ r ≤ mult)
- `dim::Int`: dim(κ)，the total irrep dimension
- `mult::Int`: total number of physical subchannels for fixed κ
- `J_tuple::Tuple`: total isospin of each species subsystem
- `coupling_path::Tuple`: intermediate isospins in a left-associated coupling scheme (J₁₂, J₁₂₃, ...)
- `multiplicity_tuple::Tuple`: internal Schur–Weyl multiplicity-copy index for each species
"""
struct IsospinSubChannel
    κ
    r::Int
    dim::Int
    mult::Int
    J_tuple::Tuple
    coupling_path::Tuple
    multiplicity_tuple::Tuple
end

"""A stable model-space exclusion: all `r` at `kappa`, or one exact `(kappa,r)`."""
struct SubchannelExclusion
    channel_name::String
    kappa
    r::Union{Nothing,Int}
end

Base.:(==)(a::SubchannelExclusion, b::SubchannelExclusion) =
    a.channel_name == b.channel_name && a.kappa == b.kappa && a.r == b.r
Base.hash(x::SubchannelExclusion, h::UInt) = hash((x.channel_name, x.kappa, x.r), h)

function _is_subchannel_excluded(exclusions, channel_name::AbstractString,
                                 sub::IsospinSubChannel)
    return any(ex -> ex.channel_name == channel_name && ex.kappa == sub.κ &&
                     (ex.r === nothing || ex.r == sub.r), exclusions)
end

function _active_subchannels(ch::FockChannel, I::Rational{Int}, exclusions)
    return [sub for sub in get_isospin_subchannels(ch, I)
            if !_is_subchannel_excluded(exclusions, ch.name, sub)]
end

"""
    get_isospin_subchannels(ch::FockChannel, I::Rational{Int}) -> Vector{IsospinSubChannel}

Return all isospin subchannels of this channel at total isospin `I`.
Single-species channels use `isospin_decomposition`; multi-species channels use `multi_isospin_decomposition`.
"""
function get_isospin_subchannels(ch::FockChannel, I::Rational{Int})
    n_threehalf = sum((ch.species[s] for s in eachindex(ch.species)
                       if ch.isospins[s] == 3//2); init=0)
    n_threehalf <= 2 || throw(ArgumentError(
        "Fock channel \"$(ch.name)\" contains $n_threehalf particles with I=3/2;" *
        "at most two particles with I=3/2 are currently supported in each FockChannel"))
    if length(ch.species) != 1
        return _multi_species_subchannels(ch, I)
    end
    # Single species: preserve the original decomposition logic.
    N_spec = ch.species[1]
    j_spec = ch.isospins[1]
    decomp = isospin_decomposition(N_spec, j_spec)
    records = NamedTuple[]
    for (J, κ, mult) in decomp.entries
        J == I || continue
        dim_κ = get_SN_irrep_dim(N_spec, κ)
        for μ in 1:mult
            push!(records, (κ=κ, dim=dim_κ, J_tuple=(J,),
                            coupling_path=(), multiplicity_tuple=(μ,)))
        end
    end
    return _number_subchannels(records)
end

function _multi_species_subchannels(ch::FockChannel, I::Rational{Int})
    decomp = multi_isospin_decomposition(ch.species, ch.isospins, I)
    records = NamedTuple[]
    for entry in decomp.entries
        dim_κ = 1
        for (s, κ_s) in enumerate(entry.κ_tuple)
            dim_κ *= get_SN_irrep_dim(ch.species[s], κ_s)
        end
        μ_ranges = map(m -> 1:m, entry.internal_multiplicities)
        for μ in Iterators.product(μ_ranges...), path in entry.coupling_paths
            push!(records, (κ=entry.κ_tuple, dim=dim_κ, J_tuple=entry.J_tuple,
                            coupling_path=path, multiplicity_tuple=Tuple(μ)))
        end
    end
    return _number_subchannels(records)
end

function _number_subchannels(records::Vector{<:NamedTuple})
    totals = Dict{Any,Int}()
    for record in records
        totals[record.κ] = get(totals, record.κ, 0) + 1
    end
    next_r = Dict{Any,Int}()
    sub_channels = IsospinSubChannel[]
    for record in records
        r = get(next_r, record.κ, 0) + 1
        next_r[record.κ] = r
        push!(sub_channels, IsospinSubChannel(
            record.κ, r, record.dim, totals[record.κ],
            record.J_tuple, record.coupling_path, record.multiplicity_tuple))
    end
    return sub_channels
end

# ============ FockSystem ============

"""
    FockSystem(d, Ncut, channels, L, a, I; Ncut_channel=nothing)

A multi-channel Fock system.

# Arguments
- `d::Momentum`: total momentum (global)
- `Ncut::Int`: global momentum cutoff; overridden when `Ncut_channel` is provided
- `channels::Vector{FockChannel}`: Fock channels
- `L::Int`: finite-volume extent (lattice units; positive integer)
- `a::Float64`: lattice spacing (fm)
- `I::Rational{Int}`: Total isospin (global conserved quantity)
- `Ncut_channel`: optional per-channel Ncut vector, whose length equals the number of channels
"""
struct FockSystem
    d::Momentum
    Ncut::Int
    Ncut_channel::Union{Nothing, Vector{Int}}
    channels::Vector{FockChannel}
    L::Int
    a::Float64
    I::Rational{Int}
    selected_irreps::Vector{String}

    function FockSystem(d, Ncut::Int, channels::Vector{FockChannel},
                        L::Int, a::Real, I, selected_irreps::Vector{String};
                        Ncut_channel::Union{Nothing, Vector{Int}}=nothing)
        Ncut >= 1 || throw(ArgumentError("Ncut must be ≥ 1"))
        L > 0 || throw(ArgumentError("L must be a positive integer"))
        a > 0 || throw(ArgumentError("lattice spacing a must be > 0"))
        length(selected_irreps) > 0 || throw(ArgumentError("at least one irrep must be selected"))
        for irr in selected_irreps
            all(x -> isascii(x), irr) || throw(ArgumentError("invalid irrep name: $irr"))
        end
        I_r = Rational{Int}(I)
        if Ncut_channel !== nothing
            length(Ncut_channel) == length(channels) || throw(ArgumentError(
                "Ncut_channel length must equal channels length"))
            all(x -> x >= 1, Ncut_channel) || throw(ArgumentError("each channel Ncut must be ≥ 1"))
        end
        d_vec = _to_momentum(d)
        new(d_vec, Ncut, Ncut_channel, channels, L, Float64(a), I_r, selected_irreps)
    end
end

get_total_N(sys::FockSystem) = sum(ch.N for ch in sys.channels)

"""
    get_Ncut(sys::FockSystem, ch_idx::Int)

Return the effective `Ncut` of channel `ch_idx`.
"""
function get_Ncut(sys::FockSystem, ch_idx::Int)
    if sys.Ncut_channel !== nothing
        return sys.Ncut_channel[ch_idx]
    else
        return sys.Ncut
    end
end

# ============ Interactive construction ============

"""
    setup_fock_system()

Interactively construct a `FockSystem` in the REPL, with prompts for total momentum, cutoffs, and channel data.
"""
function setup_fock_system()
    println("="^50)
    println("Fock-space setup")
    println("="^50)

    # Total momentum
    print("\nTotal momentum (format: nx ny nz; default: 0 0 0): ")
    d_input = strip(readline())
    if isempty(d_input)
        d = D000
    else
        parts = parse.(Int, split(d_input))
        length(parts) == 3 || throw(ArgumentError("total momentum must contain three integers"))
        d = Momentum(parts...)
    end
    println("  Total momentum d = $d")

    # Ncut policy
    print("\nUse a global Ncut? (y/n; default: y): ")
    use_global = strip(readline())
    use_global = isempty(use_global) || lowercase(use_global)[1] == 'y'

    Ncut_global = 0
    if use_global
        print("Global Ncut (lattice units; |n|² cutoff): ")
        Ncut_global = parse(Int, readline())
        Ncut_global >= 1 || throw(ArgumentError("Ncut must be ≥ 1"))
    end

    # Finite-volume parameters
    print("\nFinite-volume extent L (lattice units; positive integer): ")
    L = parse(Int, readline())
    L > 0 || throw(ArgumentError("L must be a positive integer"))

    print("Lattice spacing a (fm): ")
    a = parse(Float64, readline())
    a > 0 || throw(ArgumentError("lattice spacing a must be > 0"))

    # Total isospin (global conserved quantity)
    print("\nTotal isospin I (for example 0, 1/2, 1, 3/2, 2): ")
    I = _parse_rational(readline())

    # Number of channels
    print("\nNumber of Fock channels: ")
    num_fock = parse(Int, readline())
    num_fock >= 1 || throw(ArgumentError("at least one Fock channel is required"))

    channels = FockChannel[]
    Ncut_channel = use_global ? nothing : Int[]

    for i in 1:num_fock
        println("\n--- Channel $i ---")
        print("  Channel name (quotes optional; for example \"ππN\"): ")
        name_input = strip(readline())
        name = replace(name_input, '"' => "")
        isempty(name) && throw(ArgumentError("channel name must not be empty"))

        print("  Number of particle species: ")
        n_species = parse(Int, readline())
        n_species >= 1 || throw(ArgumentError("the number of particle species must be ≥ 1"))

        species     = Int[]
        ptypes      = Symbol[]
        masses      = Float64[]
        spins       = Rational{Int}[]
        isospins    = Rational{Int}[]
        etas        = Float64[]

        for s in 1:n_species
            println("  Particle species $s:")
            print("    Number of particles: ")
            push!(species, parse(Int, readline()))
            print("    Mass (MeV): ")
            push!(masses, parse(Float64, readline()))
            print("    Type (boson/fermion): ")
            pt = Symbol(lowercase(strip(readline())))
            pt in (:boson, :fermion) || throw(ArgumentError("particle type must be boson or fermion"))
            push!(ptypes, pt)
            print("    Spin j (0, 1/2, 1): ")
            push!(spins, _parse_rational(readline()))
            print("    Isospin i (0, 1/2, 1): ")
            push!(isospins, _parse_rational(readline()))
            print("    Intrinsic parity (+1/-1): ")
            push!(etas, parse(Float64, readline()))
        end

        print("    Kinetic-energy dispersion (relativistic/nonrelativistic; default: relativistic): ")
        kt_input = strip(readline())
        if isempty(kt_input) || lowercase(kt_input) == "relativistic"
            kt = relativistic
        elseif lowercase(kt_input) == "nonrelativistic"
            kt = nonrelativistic
        else
            throw(ArgumentError("dispersion must be relativistic or nonrelativistic"))
        end

        if !use_global
            print("  Ncut for this channel (|n|² cutoff): ")
            push!(Ncut_channel, parse(Int, readline()))
        end

        ch = FockChannel(name, species, ptypes, masses, spins, isospins, etas, kt)
        push!(channels, ch)

        # Show isospin-subchannel information
        subs = get_isospin_subchannels(ch, I)
        if !isempty(subs)
            println("  ✓ Channel \"$name\" (N=$(ch.N)) → " *
                    "$(length(unique(s->(s.κ, s.mult), subs))) subchannel(s)")
            for s_ch in unique(s -> (s.κ, s.mult), subs)
                dim_str = _kappa_dim_display(ch, s_ch.κ)
                println("      κ=$(s_ch.κ)  r=1:$(s_ch.mult)  dim=$dim_str")
            end
        else
            println("  ✓ Channel \"$name\" (N=$(ch.N)) added")
        end
    end

    # Determine the symmetry group (fermions require the double cover)
    has_fermion = any(ch -> isodd(sum((ch.species[i] for i in 1:length(ch.species) if ch.particle_types[i] == :fermion); init=0)), channels)
    _, group_name = group_for_momentum(d; double_cover=has_fermion)
    available_irreps = _get_irrep_list(group_name)
    println("\nSymmetry group: $group_name (has_fermion=$has_fermion)")
    println("Available irreps:")
    for (i, irr) in enumerate(available_irreps)
        println("  [$i] $irr")
    end
    print("Select required irreps (space-separated indices): ")
    irr_input = strip(readline())
    if isempty(irr_input)
        error("at least one irrep must be selected")
    end
    indices = parse.(Int, split(irr_input))
    selected_irreps = [available_irreps[i] for i in indices]
    println("  Selected: $selected_irreps")

    sys = FockSystem(d, Ncut_global, channels, L, a, I, selected_irreps;
                     Ncut_channel=Ncut_channel)

    println("\n" * "="^50)
    println("Fock-system setup complete")
    _print_system(sys)

    # Prepopulate representative-momentum/helicity caches
    println("\nPrecomputing representative momenta and helicities...")
    for (i, ch) in enumerate(channels)
        ncut_i = get_Ncut(sys, i)
        reps = cache_channel_reps!(ch.species, ch.particle_types, ncut_i, sys.d, ch.spins)
        n_hel = sum(length(get_helicity_reps(rep, ch.species, ch.particle_types, ch.spins, sys.d))
                    for rep in reps)
        println("  Channel $i \"$(ch.name)\": $(length(reps)) representative momenta, $n_hel representative helicities")
    end

    # Generate the interaction template
    template_path = _generate_potential_template(sys)
    println("\nFill in the interaction functions between channel pairs in $(template_path).")

    return sys
end

function _parse_rational(s::AbstractString)
    s = strip(s)
    if '/' in s
        parts = split(s, '/')
        length(parts) == 2 || throw(ArgumentError("cannot parse rational number: $s"))
        return parse(Int, parts[1]) // parse(Int, parts[2])
    else
        return parse(Int, s) // 1
    end
end

function _get_irrep_list(group_name::Symbol)
    if group_name == :Oh
        return copy(OH_IRREP_NAMES)
    elseif group_name == :Oh2
        return copy(OH2_IRREP_NAMES)
    elseif haskey(LG_IRREP_NAMES, group_name)
        return copy(LG_IRREP_NAMES[group_name])
    else
        error("unknown symmetry group: $group_name")
    end
end

function _kappa_dim(ch::FockChannel, κ)
    if κ isa String
        return get_SN_irrep_dim(ch.N, κ)
    end
    # multi-species: tuple of S_N irreps
    dim = 1
    for (sidx, κ_s) in enumerate(κ)
        dim *= get_SN_irrep_dim(ch.species[sidx], κ_s)
    end
    return dim
end

_kappa_dim_display(ch::FockChannel, κ) = string(_kappa_dim(ch, κ))

function _print_system(sys::FockSystem)
    println("  Total momentum d = $(sys.d)")
    println("  L = $(sys.L), a = $(sys.a) fm")
    println("  Total isospin I = $(sys.I)")
    println("  Selected irreps: $(sys.selected_irreps)")
    nc = sys.Ncut_channel
    for (i, ch) in enumerate(sys.channels)
        ncut_i = nc !== nothing ? nc[i] : sys.Ncut
        subs = get_isospin_subchannels(ch, sys.I)
        n_sub = length(unique(s -> (s.κ, s.mult), subs))
        println("  Channel $i: $(ch.name)  N=$(ch.N)  Ncut=$ncut_i  ($n_sub isospin subchannel(s))  $(ch.kinetic_type)")
        for s in 1:length(ch.species)
            println("    Particle species $s: $(ch.species[s]) × ($(ch.particle_types[s]), " *
                    "j=$(ch.spins[s]), I=$(ch.isospins[s]), η=$(ch.etas[s]), m=$(ch.masses[s]) MeV)")
        end
    end
end

# ============ Generate the interaction template ============

function _generate_potential_template(sys::FockSystem, output_file::String="potential_defs.jl")
    path = output_file
    io = open(path, "w")

    println(io, "# ============================================================")
    println(io, "# Interaction potential V -- user-defined matrix elements")
    println(io, "# ============================================================")
    println(io, "#")
    println(io, "# Usage:")
    println(io, "#   1. Edit the MyParams struct below with your LEC / cutoff params.")
    println(io, "#      Default values are used when you call MyParams().")
    println(io, "#")
    println(io, "#       p0 = MyParams()                       # all defaults")
    println(io, "#       p0 = MyParams(; C0=2.0)               # override C0")
    println(io, "#       x  = to_vector(p0)                    # -> Vector{Float64}")
    println(io, "#       names = param_names(MyParams)         # -> [\"C0\", \"C1\", ...]")
    println(io, "#")
    println(io, "#   2. Fill in the my_V_αβ functions below.")
    println(io, "#      They receive `params::MyParams` as the last argument.")
    println(io, "#")
    println(io, "# System summary:")
    println(io, "#   total d = $(sys.d)")
    println(io, "#   total I = $(sys.I)")
    println(io, "#   Ncut     = $(sys.Ncut)")
    println(io, "#   L        = $(sys.L)")
    println(io, "#   a        = $(sys.a) fm")
    println(io, "#   channels = $(length(sys.channels))")
    for (i, ch) in enumerate(sys.channels)
        println(io, "#     $i: \"$(ch.name)\" N=$(ch.N) species=$(ch.species)")
    end
    println(io, "# ============================================================")
    println(io)
    println(io, "using NPHFforFVE")
    println(io)
    println(io, "# ============ Parameter struct ============")
    println(io, "# Add your LECs, cutoffs etc. below. Default values are used")
    println(io, "# when no keyword argument is given.")
    println(io, "@params struct MyParams")
    println(io, "    # Examples (uncomment and edit):")
    println(io, "    # C0  = 1.0    # leading-order contact")
    println(io, "    # C1  = 0.5    # NLO contact")
    println(io, "    # Λ   = 4.0    # cutoff (1/fm)")
    println(io, "end")
    println(io)
    println(io, "# ============ V matrix elements ============")
    println(io, "#")
    println(io, "# Each function receives integer momentum vectors nA, nB.")
    println(io, "# Physical momenta pAi, pBi = (2π/L) * n are auto-generated.")
    println(io, "# For delta-function conversion:  δ^3(p'-p) → (L/(2π))^3 · δ_{n',n}")
    println(io, "#   use: if nA[i] == nB[j]  for Kronecker delta")
    println(io, "# L = $(sys.L),  (L/(2π))^3 = $((sys.L/(2π))^3)")
    println(io)

    I = sys.I
    n_ch = length(sys.channels)
    for α in 1:n_ch
        ch_α = sys.channels[α]
        subs_α = get_isospin_subchannels(ch_α, I)
        N_α = ch_α.N

        for β in 1:n_ch
            ch_β = sys.channels[β]
            subs_β = get_isospin_subchannels(ch_β, I)
            N_β = ch_β.N

            multi_α = length(ch_α.species) > 1
            multi_β = length(ch_β.species) > 1
            has_κ = !isempty(subs_α) || !isempty(subs_β)
            if has_κ
                if multi_α || multi_β
                    push!(sig_keys, "kapA", "kapB")
                else
                    push!(sig_keys, "kapA::String", "kapB::String")
                end
                push!(sig_keys, "rA::Int", "rB::Int")
            end

            sig_str = join(["nA::NTuple{$N_α,Momentum}",
                            "nB::NTuple{$N_β,Momentum}",
                            "sp::NTuple{$N_α,Rational{Int}}",
                            "s::NTuple{$N_β,Rational{Int}}",
                            sig_keys...,
                            "params::MyParams"], ", ")

            println(io, "# Channel $α \"$(ch_α.name)\" <- Channel $β \"$(ch_β.name)\"")
            println(io, "function my_V_$(α)$(β)($sig_str)")
            _gen_physical_momenta(io, sys.L, N_α, "A")
            _gen_physical_momenta(io, sys.L, N_β, "B")

            if multi_α || multi_β
                println(io, "    # TODO: multi-species channel -- auto isospin sub-channel branches not yet supported")
                println(io, "    return 0.0")
            else
                same_group = _same_sn_group(ch_α, ch_β)
                if same_group
                    _gen_same_group_branches(io, subs_α, subs_β)
                else
                    _gen_diff_group_branches(io, subs_α, subs_β)
                end
            end
            println(io, "end")
            println(io)
        end
    end

    close(io)
    return path
end

# ============ Branch-generation helpers ============

function _gen_physical_momenta(io, L::Int, N::Int, label::String)
    pis = join(["p$(label)$i = (2π/$L) .* n$label[$i]" for i in 1:N], "; ")
    println(io, "    $pis")
end

_same_sn_group(ch_α::FockChannel, ch_β::FockChannel) = ch_α.species == ch_β.species

function _need_branch(subs::Vector{IsospinSubChannel}, field::Symbol)
    vals = Set{Any}()
    for s in subs
        push!(vals, field == :κ ? s.κ : s.r)
    end
    return length(vals) > 1
end

function _gen_same_group_branches(io, subs_α, subs_β)
    α_uniq = unique(s -> (s.κ, s.r), subs_α)
    β_uniq = unique(s -> (s.κ, s.r), subs_β)

    need_κ = _need_branch(subs_α, :κ)
    need_r = _need_branch(subs_α, :r) || _need_branch(subs_β, :r)

    println(io, "    # Same permutation group: kappa-diagonal; return the complete carrier-space matrix")
    println(io, "    kapA == kapB || return 0.0")
    println(io)

    if !need_κ && !need_r
        s_α = α_uniq[1]
        println(io, "    # <$(s_α.κ),r=$(s_α.r)||V||$(s_α.κ),r=$(s_α.r)>")
        println(io, "    return 0.0")
        return
    end

    first = true
    for s_α in α_uniq, s_β in β_uniq
        s_α.κ == s_β.κ || continue

        conds = String[]
        need_κ && push!(conds, "kapA == \"$(s_α.κ)\"")
        need_r && push!(conds, "rA == $(s_α.r) && rB == $(s_β.r)")

        comment = "<$(s_α.κ),r=$(s_α.r)||V||$(s_β.κ),r=$(s_β.r)>"

        keyword = first ? "if" : "elseif"
        println(io, "    $(keyword) $(join(conds, " && "))")
        println(io, "        # $comment")
        println(io, "        return 0.0")
        first = false
    end
    println(io, "    end")
    println(io, "    return 0.0  # fallback")
end

function _gen_diff_group_branches(io, subs_α, subs_β)
    need_κ = _need_branch(subs_α, :κ) || _need_branch(subs_β, :κ)
    need_r = _need_branch(subs_α, :r) || _need_branch(subs_β, :r)
    any_branch = need_κ || need_r

    if !any_branch
        s_α = subs_α[1]
        s_β = subs_β[1]
        println(io, "    # <$(s_α.κ),r=$(s_α.r)| <- |$(s_β.κ),r=$(s_β.r)>")
        println(io, "    return 0.0")
        return
    end

    first = true
    for s_α in subs_α, s_β in subs_β
        conds = String[]
        need_κ && push!(conds, "kapA == \"$(s_α.κ)\" && kapB == \"$(s_β.κ)\"")
        need_r && push!(conds, "rA == $(s_α.r) && rB == $(s_β.r)")

        keyword = first ? "if" : "elseif"
        println(io, "    $(keyword) $(join(conds, " && "))")
        println(io, "        # <$(s_α.κ),r=$(s_α.r)| <- |$(s_β.κ),r=$(s_β.r)>")
        println(io, "        return 0.0")
        first = false
    end
    println(io, "    end")
    println(io, "    return 0.0  # fallback")
end

end # module FockSpace
