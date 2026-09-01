# ============================================================
# UserAPI — high-level user-facing API
#
# Workflow:
#   proj = Project(d, I, channels, Ncuts)        # Stage A
#   idx = add_config!(proj, L, a, irreps, n)      # Stage B (repeat)
#   ... user supplies V_func, @params ...
#   result = compute!(proj, V_func, params)        # Stage D
#   result[idx]["T1-"]                             # access values
# ============================================================

import .FockSpace: FockSystem, FockChannel, get_Ncut

"""
    Config

Configuration specified by one call to `add_config!`.

Fields:
- `L::Int`: lattice extent
- `a::Float64`: lattice spacing in fm
- `irreps`: irreducible representations to compute
- `n_levels`: requested number of lowest levels for each irrep, in the same order as `irreps`
"""
struct Config
    L::Int
    a::Float64
    irreps::Vector{String}
    n_levels::Vector{Int}
end

"""
    Project

A complete finite-volume spectrum project.

After construction, add `(L,a,Γ,n)` configurations one by one with `add_config!`, then call `compute!` to calculate all requested spectra.

Fields:
- `d::Momentum`: total momentum
- `I::Rational{Int}`: total isospin
- `channels::Vector{FockChannel}`: Fock channels
- `Ncuts::Vector{Int}`: `Ncut` for each channel; the value for an `N=1` channel is ignored
- `configs::Vector{Config}`: configurations added with `add_config!`
"""
mutable struct Project
    d::Momentum
    I::Rational{Int}
    channels::Vector{FockChannel}
    Ncuts::Vector{Int}
    configs::Vector{Config}
    V_basis::Symbol
    # Model-space exclusions; physical r labels remain stable after filtering.
    exclude_subchannels::Vector{SubchannelExclusion}
end

"""
    Project(d, I, channels, Ncuts) -> Project

Create a project.

# Arguments
- `d`: total momentum (`Momentum` or a three-tuple)
- `I`: total isospin (`Rational{Int}`)
- `channels`: Fock channels
- `Ncuts`: momentum cutoff for each channel, with the same length as `channels`. The `Ncut` of an `N=1` channel is ignored and may be set to zero.

# Example
```julia
ch_rho  = FockChannel("rho",  [1], [:boson], [800.0], [1//1], [1//1], [-1.0], relativistic)
ch_pipi = FockChannel("pipi", [2], [:boson], [140.0], [0//1], [1//1], [-1.0], relativistic)
proj = Project(Momentum(0,0,0), 1//1, [ch_rho, ch_pipi], [0, 8])
```
"""
function Project(d, I::Rational{Int}, channels::Vector{FockChannel}, Ncuts::Vector{Int};
                  V_basis::Symbol=:canonical,
                  exclude_subchannels::AbstractVector=Any[])
    V_basis in (:canonical, :helicity) ||
        throw(ArgumentError("V_basis must be :canonical or :helicity, got :$V_basis"))
    n_ch = length(channels)
    length(Ncuts) == n_ch ||
        throw(ArgumentError("Ncuts has length $(length(Ncuts)), but channels has length $n_ch"))

    for i in 1:n_ch
        n = Ncuts[i]
        if channels[i].N > 1 && n < 1
            throw(ArgumentError("channel \"$(channels[i].name)\" (N=$(channels[i].N)) requires Ncut ≥ 1, got $n"))
        end
    end

    d_mom = d isa Momentum ? d : Momentum(d...)
    proj = Project(d_mom, I, channels, Ncuts, Config[], V_basis,
                   SubchannelExclusion[])
    for rule in exclude_subchannels
        rule isa Tuple || throw(ArgumentError(
            "exclude_subchannels entries must be (channel_name, kappa) or (channel_name, kappa, r)"))
        if length(rule) == 2
            exclude_subchannel!(proj, rule[1], rule[2])
        elseif length(rule) == 3
            exclude_subchannel!(proj, rule[1], rule[2], rule[3])
        else
            throw(ArgumentError(
                "exclude_subchannels entries must have length 2 or 3, got $(length(rule))"))
        end
    end
    return proj
end

"""
    add_config!(proj::Project, L::Int, a, irreps, n_levels) -> Int

Add one `(L,a,Γ,n)` configuration and return its one-based index, which can subsequently be used as `result[idx]`.

# Arguments
- `L`: finite-volume lattice extent
- `a`: lattice spacing in fm
- `irreps`: irreducible representations of `O_h` to compute
- `n_levels`: requested number of lowest levels for each irrep, with the same length as `irreps`

# Returns
The one-based configuration index for accessing the result of `compute!`.

# Example
```julia
idx1 = add_config!(proj, 48, 0.1, ["T1-","A2-"], [5,1])
idx2 = add_config!(proj, 64, 0.08, ["T1-"], [3])
```
Configurations with the same `(L,a)` automatically share a geometry cache in `compute!`; no manual cache management is required.
"""
function add_config!(proj::Project, L::Int, a::Real,
                     irreps::Vector{String}, n_levels::Vector{Int})
    L > 0 || throw(ArgumentError("L must be > 0, got $L"))
    a > 0 || throw(ArgumentError("a must be > 0, got $a"))
    length(irreps) == length(n_levels) ||
        throw(ArgumentError("irreps and n_levels must have the same length"))

    for Γ in irreps
        valid = Γ in OH_IRREP_NAMES || Γ in OH2_IRREP_NAMES
        if !valid && proj.d != D000
            has_fermion = any(ch -> isodd(sum((ch.species[i] for i in 1:length(ch.species) if ch.particle_types[i] == :fermion); init=0)), proj.channels)
            _, group_name = group_for_momentum(proj.d; double_cover=has_fermion)
            if haskey(LG_IRREP_NAMES, group_name)
                valid = Γ in LG_IRREP_NAMES[group_name]
            end
        end
        valid || throw(ArgumentError("invalid irreducible representation: $Γ"))
    end

    push!(proj.configs, Config(L, Float64(a), irreps, n_levels))
    return length(proj.configs)
end

"""
    exclude_subchannel!(proj::Project, chan_name::String, kappa[, r])

Exclude an entire `kappa` subspace from the calculation, or only one physical subchannel `(kappa,r)`.
The type of `kappa` matches the subchannel label `κ`:
- single-species channel: `String`, for example `"[4]"`
- multiple-species channel: `NTuple{K,String}`, for example `("[1]", "[1]")`
"""
function _matching_project_channel(proj::Project, chan_name::AbstractString)
    matches = findall(ch -> ch.name == chan_name, proj.channels)
    isempty(matches) && throw(ArgumentError("Fock channel \"$chan_name\" is not in the Project"))
    length(matches) == 1 || throw(ArgumentError(
        "Fock channel name \"$chan_name\" is not unique in the Project, so the exclusion target is ambiguous"))
    return proj.channels[only(matches)]
end

function _validate_subchannel_target(proj::Project, chan_name::AbstractString,
                                     kappa, r::Union{Nothing,Integer})
    ch = _matching_project_channel(proj, chan_name)
    subs = get_isospin_subchannels(ch, proj.I)
    any(sub -> sub.κ == kappa, subs) || throw(ArgumentError(
        "Fock channel \"$chan_name\" has no κ=$(repr(kappa)) at I=$(proj.I)"))
    if r !== nothing
        r > 0 || throw(ArgumentError("subchannel r must be positive, got $r"))
        any(sub -> sub.κ == kappa && sub.r == r, subs) || throw(ArgumentError(
            "Fock channel \"$chan_name\", κ=$(repr(kappa)), has no r=$r at I=$(proj.I)"))
    end
    return ch, subs
end

function _push_subchannel_exclusion!(proj::Project, chan_name::AbstractString,
                                     kappa, r::Union{Nothing,Integer})
    _validate_subchannel_target(proj, chan_name, kappa, r)
    rule = SubchannelExclusion(String(chan_name), kappa,
                               r === nothing ? nothing : Int(r))
    if r !== nothing && any(ex -> ex.channel_name == chan_name &&
                                  ex.kappa == kappa && ex.r === nothing,
                            proj.exclude_subchannels)
        return proj
    end
    candidate = copy(proj.exclude_subchannels)
    if r === nothing
        filter!(ex -> !(ex.channel_name == chan_name && ex.kappa == kappa), candidate)
    end
    rule in candidate || push!(candidate, rule)
    active_total = sum(length(_active_subchannels(ch, proj.I, candidate))
                       for ch in proj.channels)
    if active_total == 0
        throw(ArgumentError("cannot exclude every isospin subchannel from a Project"))
    end
    empty!(proj.exclude_subchannels)
    append!(proj.exclude_subchannels, candidate)
    return proj
end

exclude_subchannel!(proj::Project, chan_name::String, kappa) =
    _push_subchannel_exclusion!(proj, chan_name, kappa, nothing)

exclude_subchannel!(proj::Project, chan_name::String, kappa, r::Integer) =
    _push_subchannel_exclusion!(proj, chan_name, kappa, r)

function include_subchannel!(proj::Project, chan_name::String, kappa)
    filter!(ex -> !(ex.channel_name == chan_name && ex.kappa == kappa &&
                    ex.r === nothing), proj.exclude_subchannels)
    return proj
end

function include_subchannel!(proj::Project, chan_name::String, kappa, r::Integer)
    filter!(ex -> !(ex.channel_name == chan_name && ex.kappa == kappa && ex.r == r),
            proj.exclude_subchannels)
    return proj
end

# ============================================================
# Spectrum evaluation and results
# ============================================================

"""
    ProjectResult

Return type of `compute!`. Index it to access the spectrum of each `Config`.

# Access
- `result[idx]` → `Dict{String, Vector{Float64}}`, the spectrum of configuration `idx`
- `length(result)` → total number of configurations
"""
struct ProjectRunInfo
    project::Project
    params
    potential::String
    backend::Symbol
    preparation::Symbol
    channel_filter::String
    entry_filter::String
    validate_hermitian::Bool
    channel_decomp::Bool
end

struct ProjectResult
    configs::Vector{Config}
    spectra::Vector{Dict{String, Vector{Float64}}}
    # channel decomposition (only populated when compute! called with channel_decomp=true)
    # [config_idx][Gamma] => Vector{Dict{String, Float64}} (one dict per eigenvalue)
    channel_decomp::Vector{Dict{String, Vector{Dict{String, Float64}}}}
    run_info::Union{Nothing,ProjectRunInfo}
end

ProjectResult(configs, spectra, channel_decomp) =
    ProjectResult(configs, spectra, channel_decomp, nothing)

function _callable_label(f)
    f === nothing && return "none"
    return try
        string(parentmodule(f), ".", nameof(f))
    catch
        string(typeof(f))
    end
end

function _run_info(proj::Project, params, V_func, backend::Symbol,
                   preparation::Symbol; channel_filter=nothing,
                   entry_filter=nothing, validate_hermitian::Bool=false,
                   channel_decomp::Bool=false)
    return ProjectRunInfo(
        deepcopy(proj), deepcopy(params), _callable_label(V_func), backend,
        preparation, _callable_label(channel_filter),
        _callable_label(entry_filter), validate_hermitian, channel_decomp)
end

"""Parameter-independent geometry prepared for repeated spectrum evaluations."""
struct PreparedSpectrumProject
    project::Project
    backend::Symbol
    groups::Dict{Tuple{Int,Float64}, SystemBasis}
end

"""
    prepare_spectrum(proj; backend=:factorized) -> PreparedSpectrumProject

Prepare the geometry, basis states, and symmetry projections that do not depend
on interaction parameters. The returned object owns a snapshot of `proj` and
can be reused for arbitrary (not necessarily affine) parameter dependence.

Changing the original `Project` after preparation does not affect this object.
Prepared objects are mutable working state during `compute!` and must not be
evaluated concurrently from multiple tasks.
"""
function prepare_spectrum(proj::Project; backend::Symbol=:factorized)
    isempty(proj.configs) &&
        throw(ArgumentError("add at least one configuration with add_config! before computing"))
    backend in (:complete_matrix, :projected_blocks, :factorized) ||
        throw(ArgumentError(
            "backend must be :complete_matrix, :projected_blocks, or :factorized"))

    prepared_project = deepcopy(proj)
    n_ch = length(prepared_project.channels)
    ncuts_sys = Int[prepared_project.Ncuts[i] > 0 ? prepared_project.Ncuts[i] : 1
                    for i in 1:n_ch]

    config_groups = Dict{Tuple{Int,Float64}, Vector{Int}}()
    for (i, cfg) in enumerate(prepared_project.configs)
        push!(get!(Vector{Int}, config_groups, (cfg.L, cfg.a)), i)
    end

    groups = Dict{Tuple{Int,Float64}, SystemBasis}()
    for ((L, a), config_indices) in config_groups
        all_irreps = String[]
        for idx in config_indices, Gamma in prepared_project.configs[idx].irreps
            Gamma in all_irreps || push!(all_irreps, Gamma)
        end
        sys = FockSystem(prepared_project.d, 1, prepared_project.channels,
                         L, a, prepared_project.I, all_irreps;
                         Ncut_channel=ncuts_sys)
        groups[(L, a)] = SystemBasis(
            sys; exclude_subchannels=prepared_project.exclude_subchannels)
    end
    return PreparedSpectrumProject(prepared_project, backend, groups)
end

"""
    compute!(prepared::PreparedSpectrumProject, V_func, params; kwargs...)

Evaluate spectra while reusing the parameter-independent geometry created by
[`prepare_spectrum`](@ref). The interaction blocks are rebuilt for every
parameter point. This method is serial with respect to a given `prepared`
object because it updates the cached basis interaction blocks in place.
"""
function compute!(prepared::PreparedSpectrumProject, V_func, params;
                  channel_filter=nothing, entry_filter=nothing,
                  verbose::Bool=false, channel_decomp::Bool=false,
                  validate_hermitian::Bool=false)
    proj = prepared.project
    n_configs = length(proj.configs)
    spectra = Vector{Dict{String, Vector{Float64}}}(undef, n_configs)
    decomp = [Dict{String, Vector{Dict{String, Float64}}}()
              for _ in 1:n_configs]

    config_groups = Dict{Tuple{Int,Float64}, Vector{Int}}()
    for (i, cfg) in enumerate(proj.configs)
        push!(get!(Vector{Int}, config_groups, (cfg.L, cfg.a)), i)
    end

    for (key, config_indices) in config_groups
        basis = prepared.groups[key]
        build_V_hel_blocks!(basis, V_func, params;
                            V_basis=proj.V_basis,
                            channel_filter=channel_filter,
                            entry_filter=entry_filter,
                            validate_hermitian=validate_hermitian)
        verbose && println(basis)

        for idx in config_indices
            cfg = proj.configs[idx]
            n_levels = Dict(cfg.irreps[i] => cfg.n_levels[i]
                            for i in eachindex(cfg.irreps))
            if prepared.backend == :projected_blocks
                if channel_decomp
                    evals, evecs = compute_spectrum_eigs(
                        basis; n_levels=n_levels, return_vectors=true)
                    spectra[idx] = evals
                    for (Gamma, vectors) in evecs
                        decomp[idx][Gamma] =
                            channel_decomposition(basis, Gamma, vectors)
                    end
                else
                    spectra[idx] = compute_spectrum_eigs(
                        basis; n_levels=n_levels)
                end
            elseif prepared.backend == :factorized
                if channel_decomp
                    evals, evecs = compute_spectrum_factorized(
                        basis; n_levels=n_levels, return_vectors=true)
                    spectra[idx] = evals
                    for (Gamma, vectors) in evecs
                        decomp[idx][Gamma] =
                            channel_decomposition(basis, Gamma, vectors)
                    end
                else
                    spectra[idx] = compute_spectrum_factorized(
                        basis; n_levels=n_levels)
                end
            else
                if channel_decomp
                    evals, evecs = compute_spectrum(
                        basis; n_levels=n_levels, return_vectors=true)
                    spectra[idx] = evals
                    for (Gamma, vectors) in evecs
                        decomp[idx][Gamma] =
                            channel_decomposition(basis, Gamma, vectors)
                    end
                else
                    spectra[idx] = compute_spectrum(basis; n_levels=n_levels)
                end
            end
        end
    end
    info = _run_info(proj, params, V_func, prepared.backend,
                     :prepared_spectrum;
                     channel_filter=channel_filter,
                     entry_filter=entry_filter,
                     validate_hermitian=validate_hermitian,
                     channel_decomp=channel_decomp)
    return ProjectResult(info.project.configs, spectra, decomp, info)
end

const _AffineProjectedKey = NTuple{4,Int}

"""Projected affine interaction cache for one irrep."""
struct AffineIrrepCache
    keys::Vector{_AffineProjectedKey}
    constant_blocks::Vector{Matrix{ComplexF64}}
    coefficient_blocks::Vector{Vector{Matrix{ComplexF64}}}
end

"""One `(L,a)` geometry cache produced by [`prepare_affine`](@ref)."""
struct AffineBasisCache
    basis::SystemBasis
    irreps::Dict{String,AffineIrrepCache}
end

"""
Parameter-independent data for an affine interaction. Fields referenced by
`dynamic_mass` are reserved for the kinetic energy and are not potential
coefficients.
"""
struct AffinePreparedProject{P}
    project::Project
    param_type::Type{P}
    parameter_names::Vector{String}
    potential_parameter_indices::Vector{Int}
    fixed_parameter_indices::Vector{Int}
    fixed_parameter_values::Vector{Float64}
    backend::Symbol
    groups::Dict{Tuple{Int,Float64}, AffineBasisCache}
    potential::String
    channel_filter::String
    entry_filter::String
    validate_hermitian::Bool
end

function _affine_fixed_flags(fixed, names::Vector{String})
    fixed === nothing && return falses(length(names))
    if fixed isa AbstractVector
        length(fixed) == length(names) || throw(DimensionMismatch(
            "fixed has length $(length(fixed)); expected $(length(names))"))
        all(value -> value isa Bool, fixed) || throw(ArgumentError(
            "all fixed settings must be Bool"))
        return BitVector(fixed)
    end
    fixed isa NamedTuple || fixed isa AbstractDict || throw(ArgumentError(
        "fixed must be a NamedTuple, dictionary, Bool vector, or nothing"))
    entries = Dict{String,Any}(String(key) => value
                               for (key, value) in pairs(fixed))
    unknown = setdiff(collect(keys(entries)), names)
    isempty(unknown) || throw(ArgumentError(
        "fixed contains unknown parameter names: " *
        join(sort(unknown), ", ")))
    all(value -> value isa Bool, values(entries)) || throw(ArgumentError(
        "all fixed settings must be Bool"))
    return BitVector(get(entries, name, false) for name in names)
end

function _copy_V_blocks(blocks)
    result = Dict{Tuple{Int,Int,Int,Int}, AbstractMatrix{ComplexF64}}()
    for (key, block) in blocks
        result[key] = copy(block)
    end
    return result
end

function _subtract_V_blocks(blocks, constant_blocks)
    Set(keys(blocks)) == Set(keys(constant_blocks)) ||
        throw(ArgumentError("affine probe produced a different V_hel block structure"))
    result = Dict{Tuple{Int,Int,Int,Int}, AbstractMatrix{ComplexF64}}()
    for (key, block) in blocks
        base = constant_blocks[key]
        size(block) == size(base) || throw(ArgumentError(
            "affine probe changed the size of V_hel block $key; " *
            "channel_filter and entry_filter must be parameter-independent"))
        result[key] = block - base
    end
    return result
end

function _project_affine_V_blocks(basis::SystemBasis, blocks, Gamma::String)
    empty!(basis.V_hel_blocks)
    merge!(basis.V_hel_blocks, blocks)
    _, projected = _build_hamiltonian_operator(
        basis, Gamma; include_kinetic=false)
    result = Dict{_AffineProjectedKey, Matrix{ComplexF64}}()
    for block in projected
        key = (first(block.r_rng), last(block.r_rng),
               first(block.c_rng), last(block.c_rng))
        if haskey(result, key)
            result[key] .+= block.mat
        else
            result[key] = copy(block.mat)
        end
    end
    return result
end

function _make_affine_irrep_cache(constant, coefficients)
    all_keys = Set{_AffineProjectedKey}(keys(constant))
    for blocks in coefficients
        union!(all_keys, keys(blocks))
    end
    ordered_keys = sort!(collect(all_keys))

    constant_blocks = Matrix{ComplexF64}[]
    coefficient_blocks = [Matrix{ComplexF64}[] for _ in coefficients]
    for key in ordered_keys
        nr = key[2] - key[1] + 1
        nc = key[4] - key[3] + 1
        push!(constant_blocks,
              haskey(constant, key) ? constant[key] :
              zeros(ComplexF64, nr, nc))
        for i in eachindex(coefficients)
            push!(coefficient_blocks[i],
                  haskey(coefficients[i], key) ? coefficients[i][key] :
                  zeros(ComplexF64, nr, nc))
        end
    end
    return AffineIrrepCache(
        ordered_keys, constant_blocks, coefficient_blocks)
end

"""
    prepare_affine(proj, V_func, params_template;
                   backend=:factorized, channel_filter=nothing,
                   entry_filter=nothing, fixed=nothing,
                   validate_hermitian=false) -> AffinePreparedProject

Prepare the parameter-independent interaction for
`V(params) = V0 + sum(params[i] * Vi)`.

`params_template` must be an instance of a type created by `@params`.
Normally its numeric values are ignored. Fields selected by `fixed` instead
take their constant values from `params_template` and are absorbed into
`V0`. One unit-vector probe is made only for each non-fixed potential
parameter. Pass the same `fixed` setting to the fitting backend.

In a rest-frame project, fields referenced by `dynamic_mass` are excluded from
the potential probes and are resolved afresh for the kinetic energy at every
`compute!` call. This is a user contract: the potential must not read those
fields. Moving-frame projects containing dynamic masses must use
[`prepare_spectrum`](@ref), because their boosted interaction kinematics also
depend nonlinearly on the masses.

Probes run serially because each `build_V_hel_blocks!` call already performs
its own internal parallel work. Supplied filters must be independent of the
fit-parameter values.

The coefficient matrices are projected during preparation and cached in
each requested irrep. By linearity,
`X' * (V0 + sum(pi*Vi)) * X = X'V0X + sum(pi*X'ViX)`, so evaluation does not
reconstruct the much larger helicity-space matrices. Evaluation through
`compute!(prepared, params)` is implemented separately.
"""
function prepare_affine(proj::Project, V_func, params_template;
                        backend::Symbol=:factorized,
                        channel_filter=nothing,
                        entry_filter=nothing,
                        fixed=nothing,
                        verbose::Bool=false,
                        validate_hermitian::Bool=false)
    isempty(proj.configs) &&
        throw(ArgumentError("add at least one configuration with add_config! before computing"))
    backend in (:complete_matrix, :projected_blocks, :factorized) ||
        throw(ArgumentError(
            "backend must be :complete_matrix, :projected_blocks, or :factorized"))

    P = typeof(params_template)
    names = try
        String.(param_names(P))
    catch err
        throw(ArgumentError(
            "params_template must be an instance of a type created by @params; " *
            "param_names($P) is unavailable: $(sprint(showerror, err))"))
    end
    n_params = length(names)
    n_params > 0 || throw(ArgumentError("affine preparation requires at least one parameter"))

    dynamic_names = unique(m.name for ch in proj.channels for m in ch.masses
                           if m isa DynamicMass)
    missing_names = setdiff(dynamic_names, Symbol.(names))
    isempty(missing_names) || throw(ArgumentError(
        "dynamic mass parameter fields missing from $P: " *
        join(("`$name`" for name in missing_names), ", ")))
    if proj.d != D000 && !isempty(dynamic_names)
        throw(ArgumentError(
            "prepare_affine does not support dynamic masses in a moving frame; " *
            "use prepare_spectrum instead"))
    end
    fixed_flags = _affine_fixed_flags(fixed, names)
    fixed_parameter_indices = findall(fixed_flags)
    template_values = Float64.(to_vector(params_template))
    length(template_values) == n_params || throw(DimensionMismatch(
        "params_template vector length $(length(template_values)) does not " *
        "match parameter-name length $n_params"))
    fixed_parameter_values = template_values[fixed_parameter_indices]
    potential_parameter_indices = findall(
        i -> Symbol(names[i]) ∉ dynamic_names && !fixed_flags[i], 1:n_params)

    base_values = zeros(Float64, n_params)
    base_values[fixed_parameter_indices] .= fixed_parameter_values
    base_params = try
        from_vector(P, base_values)
    catch err
        throw(ArgumentError(
            "cannot construct affine base/unit probes for $P with from_vector: " *
            sprint(showerror, err)))
    end

    prepared_project = deepcopy(proj)
    n_ch = length(prepared_project.channels)
    ncuts_sys = Int[prepared_project.Ncuts[i] > 0 ? prepared_project.Ncuts[i] : 1
                    for i in 1:n_ch]

    config_groups = Dict{Tuple{Int,Float64}, Vector{Int}}()
    for (i, cfg) in enumerate(prepared_project.configs)
        push!(get!(Vector{Int}, config_groups, (cfg.L, cfg.a)), i)
    end

    prepared_groups = Dict{Tuple{Int,Float64}, AffineBasisCache}()
    for ((L, a), config_indices) in config_groups
        all_irreps = String[]
        for idx in config_indices, Gamma in prepared_project.configs[idx].irreps
            Gamma in all_irreps || push!(all_irreps, Gamma)
        end

        sys = FockSystem(prepared_project.d, 1, prepared_project.channels,
                         L, a, prepared_project.I, all_irreps;
                         Ncut_channel=ncuts_sys)
        basis = SystemBasis(sys;
                            exclude_subchannels=prepared_project.exclude_subchannels)

        verbose && println("prepare_affine: (L=$L, a=$a), constant term")
        build_V_hel_blocks!(basis, V_func, base_params;
                            V_basis=prepared_project.V_basis,
                            channel_filter=channel_filter,
                            entry_filter=entry_filter,
                            validate_hermitian=validate_hermitian,
                            _resolve_masses=false)
        constant_blocks = _copy_V_blocks(basis.V_hel_blocks)
        projected_constant = Dict(
            Gamma => _project_affine_V_blocks(basis, constant_blocks, Gamma)
            for Gamma in all_irreps)
        projected_coefficients = Dict(
            Gamma => Vector{
                Dict{_AffineProjectedKey,Matrix{ComplexF64}}
            }(undef, length(potential_parameter_indices))
            for Gamma in all_irreps)
        probe = copy(base_values)
        for (coefficient_index, parameter_index) in enumerate(potential_parameter_indices)
            copyto!(probe, base_values)
            probe[parameter_index] = 1.0
            unit_params = from_vector(P, probe)
            verbose && println("prepare_affine: (L=$L, a=$a), coefficient $(names[parameter_index])")
            build_V_hel_blocks!(basis, V_func, unit_params;
                                V_basis=prepared_project.V_basis,
                                channel_filter=channel_filter,
                                entry_filter=entry_filter,
                                validate_hermitian=validate_hermitian,
                                _resolve_masses=false)
            coefficient_blocks =
                _subtract_V_blocks(basis.V_hel_blocks, constant_blocks)
            for Gamma in all_irreps
                projected_coefficients[Gamma][coefficient_index] =
                    _project_affine_V_blocks(
                        basis, coefficient_blocks, Gamma)
            end
        end

        empty!(basis.V_hel_blocks)
        irrep_caches = Dict(
            Gamma => _make_affine_irrep_cache(
                projected_constant[Gamma], projected_coefficients[Gamma])
            for Gamma in all_irreps)
        prepared_groups[(L, a)] = AffineBasisCache(basis, irrep_caches)
    end

    return AffinePreparedProject{P}(
        prepared_project, P, names, potential_parameter_indices,
        fixed_parameter_indices, fixed_parameter_values, backend, prepared_groups,
        _callable_label(V_func), _callable_label(channel_filter),
        _callable_label(entry_filter), validate_hermitian)
end

function _combine_affine_projected_blocks(
        cache::AffineBasisCache, values::AbstractVector,
        parameter_indices::AbstractVector{Int})
    result = Dict{String,Vector{_VBlock}}()
    for (Gamma, irrep_cache) in cache.irreps
        length(parameter_indices) == length(irrep_cache.coefficient_blocks) ||
            throw(DimensionMismatch(
                "received $(length(parameter_indices)) affine parameter indices, expected " *
                "$(length(irrep_cache.coefficient_blocks))"))
        blocks = _VBlock[]
        for (block_index, key) in enumerate(irrep_cache.keys)
            combined = copy(irrep_cache.constant_blocks[block_index])
            for (coefficient_index, parameter_index) in enumerate(parameter_indices)
                value = values[parameter_index]
                iszero(value) && continue
                combined .+= value .* irrep_cache.coefficient_blocks[
                    coefficient_index][block_index]
            end
            all(iszero, combined) && continue
            r_rng = key[1]:key[2]
            c_rng = key[3]:key[4]
            push!(blocks, _VBlock(
                r_rng, c_rng, combined, r_rng != c_rng))
        end
        result[Gamma] = blocks
    end
    return result
end

"""
    compute!(prepared::AffinePreparedProject, params;
             verbose=false, channel_decomp=false) -> ProjectResult

Evaluate spectra from data created by [`prepare_affine`](@ref). The cached
projected interaction is reconstructed as `V0 + sum(params[i] * Vi)` and
`V_func` is not called. Dynamic masses are resolved for the kinetic term on
every evaluation.
The solver backend is the one selected during preparation.

`params` must have exactly the same `@params` type used by `prepare_affine`.
The prepared object is mutable working state and must not be evaluated
concurrently from multiple tasks.
"""
function compute!(prepared::AffinePreparedProject{P}, params;
                  verbose::Bool=false,
                  channel_decomp::Bool=false) where P
    typeof(params) === P || throw(ArgumentError(
        "parameter type mismatch: prepared for $P, received $(typeof(params))"))
    values = Float64.(to_vector(params))
    length(values) == length(prepared.parameter_names) ||
        throw(DimensionMismatch(
            "parameter vector length $(length(values)) does not match prepared " *
            "length $(length(prepared.parameter_names))"))
    for (index, fixed_value) in zip(
            prepared.fixed_parameter_indices, prepared.fixed_parameter_values)
        isequal(values[index], fixed_value) || throw(ArgumentError(
            "affine parameter $(prepared.parameter_names[index]) was fixed at " *
            "$fixed_value during prepare_affine but compute! received " *
            "$(values[index])"))
    end

    proj = prepared.project
    n_configs = length(proj.configs)
    spectra = Vector{Dict{String, Vector{Float64}}}(undef, n_configs)
    decomp = [Dict{String, Vector{Dict{String, Float64}}}()
              for _ in 1:n_configs]

    config_groups = Dict{Tuple{Int,Float64}, Vector{Int}}()
    for (i, cfg) in enumerate(proj.configs)
        push!(get!(Vector{Int}, config_groups, (cfg.L, cfg.a)), i)
    end

    for (key, config_indices) in config_groups
        cache = prepared.groups[key]
        basis = cache.basis
        projected_blocks = _combine_affine_projected_blocks(
            cache, values, prepared.potential_parameter_indices)
        _resolve_basis_masses!(basis, params)
        verbose && println(basis)

        for idx in config_indices
            cfg = proj.configs[idx]
            n_levels = Dict(cfg.irreps[i] => cfg.n_levels[i]
                            for i in eachindex(cfg.irreps))

            solve_backend = prepared.backend == :complete_matrix ?
                :complete_matrix : :projected_blocks
            if channel_decomp
                evals, evecs = _compute_spectrum_preprojected(
                    basis, projected_blocks; n_levels=n_levels,
                    backend=solve_backend, return_vectors=true)
                spectra[idx] = evals
                for (Gamma, vectors) in evecs
                    decomp[idx][Gamma] =
                        channel_decomposition(basis, Gamma, vectors)
                end
            else
                spectra[idx] = _compute_spectrum_preprojected(
                    basis, projected_blocks; n_levels=n_levels,
                    backend=solve_backend)
            end
        end
    end

    info = ProjectRunInfo(
        deepcopy(proj), deepcopy(params), prepared.potential,
        prepared.backend, :prepared_affine, prepared.channel_filter,
        prepared.entry_filter, prepared.validate_hermitian, channel_decomp)
    return ProjectResult(info.project.configs, spectra, decomp, info)
end

Base.length(r::ProjectResult) = length(r.spectra)

function Base.getindex(r::ProjectResult, idx::Int)
    1 <= idx <= length(r.spectra) || throw(BoundsError(r, idx))
    return r.spectra[idx]
end

"""
    compute!(proj::Project, V_func, params;
             backend=:complete_matrix, channel_filter=nothing,
             entry_filter=nothing) -> ProjectResult

Compute spectra for all configurations. Configurations with the same `(L,a)` automatically share their geometry cache and `V_hel`; no manual cache management is required.

# Arguments
- `proj`: a Project with its channels and configurations already defined
- `V_func`: interaction function with the signature required by `build_hamiltonian_block`
- `params`: parameter object for `V_func`, created with `@params`
- `backend`: explicitly selects the Hamiltonian backend: `:complete_matrix` constructs the complete matrix and uses LAPACK; `:projected_blocks` stores projected blocks and uses Hermitian Lanczos; `:factorized` applies `Q†V_hel Q` on demand and uses Hermitian Lanczos.
- `eigs`: legacy compatibility keyword. `true` maps to `:projected_blocks` and `false` to `:complete_matrix`; use `backend` in new code.
- `channel_filter`: optional Fock-channel-pair filter with signature `channel_filter(chA, chB, params)`. Returning `false` skips the complete channel pair.
- `entry_filter`: optional matrix-element filter with the same signature as `V_func`. Returning `false` skips the matrix element and the expensive `V_func` call.
- `validate_hermitian`: defaults to `false`. For debugging, validates Hermiticity on a small sample of complete blocks. When filters are provided, it first independently checks their symmetry under final/initial exchange, then checks `V_βα ≈ V_αβ†` for the unfiltered `V_func`, so filtered elements cannot mask an interaction error.

# Returns
A `ProjectResult`. `result[idx]` gives `Dict{String, Vector{Float64}}` for configuration `idx`.

# Example
```julia
result = compute!(proj, V_rho_pipi, RhoPiPiParams(g=1.43e-5))
result[1]          # Dict("T1-" => [...], "A2-" => [...])
result[1]["T1-"]   # T1- energy-level vector
```
"""
function compute!(proj::Project, V_func, params;
                  backend::Union{Nothing,Symbol}=nothing,
                  eigs::Union{Nothing,Bool}=nothing,
                  channel_filter=nothing, entry_filter=nothing,
                  verbose::Bool=false,
                  channel_decomp::Bool=false,
                  validate_hermitian::Bool=false)
    isempty(proj.configs) &&
        throw(ArgumentError("add at least one configuration with add_config! before computing"))

    selected_backend = if backend === nothing
        eigs === nothing ? :complete_matrix :
        (eigs ? :projected_blocks : :complete_matrix)
    else
        eigs === nothing || throw(ArgumentError(
            "cannot specify both backend and legacy eigs"))
        backend
    end
    selected_backend in (:complete_matrix, :projected_blocks, :factorized) ||
        throw(ArgumentError(
            "backend must be :complete_matrix, :projected_blocks, or :factorized"))

    n_ch = length(proj.channels)
    n_configs = length(proj.configs)

    # N=1-channel Ncut is a dummy (ignored) value, ensuring FockSystem construction succeeds.
    ncuts_sys = Int[proj.Ncuts[i] > 0 ? proj.Ncuts[i] : 1 for i in 1:n_ch]

    # Group by (L,a); configurations in each group share a SystemBasis.
    groups = Dict{Tuple{Int, Float64}, Vector{Int}}()
    for (i, cfg) in enumerate(proj.configs)
        key = (cfg.L, cfg.a)
        idxs = get!(Vector{Int}, groups, key)
        push!(idxs, i)
    end

    spectra = Vector{Dict{String, Vector{Float64}}}(undef, n_configs)
    decomp = [Dict{String, Vector{Dict{String, Float64}}}()
              for _ in 1:n_configs]

    for ((L, a), config_indices) in groups
        # Union of all irreps in the group
        all_irreps = String[]
        for idx in config_indices
            for Gamma in proj.configs[idx].irreps
                Gamma in all_irreps || push!(all_irreps, Gamma)
            end
        end

        sys = FockSystem(proj.d, 1, proj.channels, L, a, proj.I, all_irreps;
                         Ncut_channel=ncuts_sys)
        basis = SystemBasis(sys; exclude_subchannels=proj.exclude_subchannels)
        build_V_hel_blocks!(basis, V_func, params;
                            V_basis=proj.V_basis,
                            channel_filter=channel_filter,
                            entry_filter=entry_filter,
                            validate_hermitian=validate_hermitian)

        verbose && println(basis)

        for idx in config_indices
            cfg = proj.configs[idx]
            n_levels = Dict(cfg.irreps[i] => cfg.n_levels[i]
                            for i in 1:length(cfg.irreps))
            if selected_backend == :projected_blocks
                if channel_decomp
                    evals, evecs = compute_spectrum_eigs(basis; n_levels=n_levels,
                                                         return_vectors=true)
                    spectra[idx] = evals
                    decomp[idx] = Dict{String, Vector{Dict{String, Float64}}}()
                    for (Gamma, V) in evecs
                        decomp[idx][Gamma] = channel_decomposition(basis, Gamma, V)
                    end
                else
                    spectra[idx] = compute_spectrum_eigs(basis; n_levels=n_levels)
                end
            elseif selected_backend == :factorized
                if channel_decomp
                    evals, evecs = compute_spectrum_factorized(
                        basis; n_levels=n_levels, return_vectors=true)
                    spectra[idx] = evals
                    decomp[idx] = Dict{String, Vector{Dict{String, Float64}}}()
                    for (Gamma, V) in evecs
                        decomp[idx][Gamma] = channel_decomposition(basis, Gamma, V)
                    end
                else
                    spectra[idx] = compute_spectrum_factorized(
                        basis; n_levels=n_levels)
                end
            else
                if channel_decomp
                    evals, evecs = compute_spectrum(basis; n_levels=n_levels,
                                                    return_vectors=true)
                    spectra[idx] = evals
                    decomp[idx] = Dict{String, Vector{Dict{String, Float64}}}()
                    for (Gamma, V) in evecs
                        decomp[idx][Gamma] = channel_decomposition(basis, Gamma, V)
                    end
                else
                    spectra[idx] = compute_spectrum(basis; n_levels=n_levels)
                end
            end
        end
    end

    info = _run_info(proj, params, V_func, selected_backend, :direct;
                     channel_filter=channel_filter,
                     entry_filter=entry_filter,
                     validate_hermitian=validate_hermitian,
                     channel_decomp=channel_decomp)
    return ProjectResult(info.project.configs, spectra, decomp, info)
end

function _write_run_info(io::IO, info::ProjectRunInfo)
    proj = info.project
    println(io, "# NPHFforFVE spectrum")
    println(io, "# parameter_type = ", typeof(info.params))
    println(io, "# parameters = ", repr(info.params))
    println(io, "# total_momentum = ", Tuple(proj.d))
    println(io, "# total_isospin = ", proj.I)
    println(io, "# V_basis = ", proj.V_basis)
    println(io, "# Ncuts = ", repr(proj.Ncuts))
    println(io, "# channel_decomp = ", info.channel_decomp)
    for (i, ch) in enumerate(proj.channels)
        println(io, "# channel[$i] = ", repr((
            name=ch.name, species=ch.species,
            particle_types=ch.particle_types, masses=ch.masses,
            spins=ch.spins, isospins=ch.isospins, etas=ch.etas,
            kinetic_type=ch.kinetic_type)))
    end
    for (i, cfg) in enumerate(proj.configs)
        println(io, "# config[$i] = ", repr((
            L=cfg.L, a=cfg.a, irreps=cfg.irreps,
            n_levels=cfg.n_levels)))
    end
end

"""
    write_spectrum(result::ProjectResult, filename) -> String

Write reproducibility metadata and one tab-separated row per energy level.
If `channel_decomp=true` was used, the final column contains Fock-channel
weights. Returns `filename`.
"""
function write_spectrum(result::ProjectResult, filename::AbstractString)
    open(filename, "w") do io
        if result.run_info === nothing
            println(io, "# NPHFforFVE spectrum")
            println(io, "# run_info = unavailable")
        else
            _write_run_info(io, result.run_info)
        end
        println(io, "config\tL\ta_fm\tL_phys_fm\tirrep\tlevel\tenergy_MeV\tchannel_weights")
        for config in eachindex(result.configs)
            cfg = result.configs[config]
            for irrep in sort!(collect(keys(result[config])))
                energies = result[config][irrep]
                decompositions = get(result.channel_decomp[config], irrep,
                                     Dict{String,Float64}[])
                for (level, energy) in enumerate(energies)
                    weights = if level <= length(decompositions)
                        join(("$name=$(decompositions[level][name])" for name in
                              sort!(collect(keys(decompositions[level])))), ";")
                    else
                        ""
                    end
                    println(io, config, '\t', cfg.L, '\t', cfg.a, '\t',
                            cfg.L * cfg.a, '\t', irrep, '\t', level, '\t',
                            energy, '\t', weights)
                end
            end
        end
    end
    return String(filename)
end

# ============================================================
# Spectrum fitting data
# ============================================================

"""Location of one sorted finite-volume energy level."""
struct SpectrumLevel
    config::Int
    irrep::String
    level::Int
    function SpectrumLevel(config::Integer, irrep::AbstractString,
                           level::Integer)
        config > 0 || throw(ArgumentError("config index must be positive"))
        level > 0 || throw(ArgumentError("level index must be positive"))
        new(Int(config), String(irrep), Int(level))
    end
end

const _SpectrumCovarianceFactor =
    LinearAlgebra.Cholesky{Float64, Matrix{Float64}}

"""Observed levels belonging to one `(config, irrep)` pair."""
struct SpectrumGroup
    config::Int
    irrep::String
    values::Vector{Float64}
    levels::Union{Nothing,Vector{Int}}
    errors::Union{Nothing,Vector{Float64}}
    covariance::Union{Nothing,Matrix{Float64}}
    covariance_factor::Union{Nothing,_SpectrumCovarianceFactor}
end

function _factor_covariance(covariance, n::Int, context::AbstractString)
    C = Matrix{Float64}(covariance)
    size(C) == (n, n) || throw(DimensionMismatch(
        "$context covariance has size $(size(C)); expected ($n, $n)"))
    all(isfinite, C) || throw(ArgumentError(
        "$context covariance must contain only finite values"))
    isapprox(C, transpose(C); rtol=1e-12, atol=1e-14) ||
        throw(ArgumentError("$context covariance must be symmetric"))
    C = Matrix(Symmetric((C + transpose(C)) / 2))
    factor = try
        cholesky(Symmetric(C); check=true)
    catch err
        err isa PosDefException || rethrow()
        throw(ArgumentError("$context covariance must be positive definite"))
    end
    return C, factor
end

function SpectrumGroup(config::Integer, irrep::AbstractString, values;
                       levels=nothing, errors=nothing, covariance=nothing)
    config > 0 || throw(ArgumentError("config index must be positive"))
    vals = Float64.(collect(values))
    isempty(vals) && throw(ArgumentError("a SpectrumGroup cannot be empty"))
    all(isfinite, vals) || throw(ArgumentError(
        "spectrum data values must be finite"))

    levs = if levels === nothing
        nothing
    else
        result = Int.(collect(levels))
        length(result) == length(vals) || throw(DimensionMismatch(
            "levels and values must have the same length"))
        all(>(0), result) || throw(ArgumentError(
            "all level indices must be positive"))
        allunique(result) || throw(ArgumentError(
            "level indices within a SpectrumGroup must be unique"))
        result
    end

    errors === nothing || covariance === nothing || throw(ArgumentError(
        "provide either errors or covariance for a SpectrumGroup, not both"))
    errs = if errors === nothing
        nothing
    else
        result = Float64.(collect(errors))
        length(result) == length(vals) || throw(DimensionMismatch(
            "errors and values must have the same length"))
        all(x -> isfinite(x) && x > 0, result) || throw(ArgumentError(
            "all spectrum errors must be finite and positive"))
        result
    end
    C, factor = covariance === nothing ? (nothing, nothing) :
        _factor_covariance(covariance, length(vals), "SpectrumGroup")
    return SpectrumGroup(Int(config), String(irrep), vals, levs, errs,
                         C, factor)
end

"""A collection of spectrum groups with local or global uncertainties."""
struct SpectrumDataset
    groups::Vector{SpectrumGroup}
    covariance::Union{Nothing,Matrix{Float64}}
    covariance_factor::Union{Nothing,_SpectrumCovarianceFactor}
end

function SpectrumDataset(groups::AbstractVector{<:SpectrumGroup};
                         covariance=nothing)
    gs = SpectrumGroup[groups...]
    isempty(gs) && throw(ArgumentError("a SpectrumDataset cannot be empty"))
    if covariance === nothing
        for (i, group) in enumerate(gs)
            (group.errors !== nothing || group.covariance !== nothing) ||
                throw(ArgumentError(
                    "SpectrumGroup $i must provide errors or covariance " *
                    "when no global covariance is supplied"))
        end
        return SpectrumDataset(gs, nothing, nothing)
    end

    for (i, group) in enumerate(gs)
        (group.errors === nothing && group.covariance === nothing) ||
            throw(ArgumentError(
                "SpectrumGroup $i cannot provide local uncertainties when " *
                "the dataset has a global covariance"))
    end
    n = sum(length(group.values) for group in gs)
    C, factor = _factor_covariance(covariance, n, "SpectrumDataset")
    return SpectrumDataset(gs, C, factor)
end

function _group_residual(result::ProjectResult, group::SpectrumGroup)
    group.config <= length(result) || throw(BoundsError(
        result.spectra, group.config))
    spectra = result[group.config]
    haskey(spectra, group.irrep) || throw(ArgumentError(
        "config $(group.config) has no spectrum for irrep $(repr(group.irrep))"))
    predicted = spectra[group.irrep]
    levels = group.levels === nothing ? collect(1:length(group.values)) :
        group.levels
    group.levels === nothing && length(predicted) == length(group.values) ||
        group.levels !== nothing || throw(DimensionMismatch(
            "config $(group.config), irrep $(group.irrep) produced " *
            "$(length(predicted)) levels but the group contains " *
            "$(length(group.values)) values"))
    maximum(levels) <= length(predicted) || throw(DimensionMismatch(
        "requested level $(maximum(levels)) for config $(group.config), " *
        "irrep $(group.irrep), but only $(length(predicted)) levels exist"))
    theory = Float64[predicted[level] for level in levels]
    all(isfinite, theory) || throw(ArgumentError(
        "predicted spectrum contains non-finite values for config " *
        "$(group.config), irrep $(group.irrep)"))
    return theory - group.values
end

"""Compute the built-in spectrum chi-squared for `result` and `data`."""
function spectrum_chisq(result::ProjectResult, data::SpectrumDataset)
    residuals = Vector{Vector{Float64}}(undef, length(data.groups))
    for (i, group) in enumerate(data.groups)
        residuals[i] = _group_residual(result, group)
    end

    chi2 = if data.covariance_factor !== nothing
        delta = reduce(vcat, residuals)
        dot(delta, data.covariance_factor \ delta)
    else
        total = 0.0
        for (group, delta) in zip(data.groups, residuals)
            total += if group.errors !== nothing
                sum(abs2, delta ./ group.errors)
            else
                dot(delta, group.covariance_factor \ delta)
            end
        end
        total
    end
    isfinite(chi2) || throw(ArgumentError("spectrum chi-squared is not finite"))
    return chi2
end

# ============================================================
# Spectrum fit objective
# ============================================================

"""
    SpectrumFitProblem(...)

A callable vector objective that converts fit coordinates back to an
`@params` object, computes a spectrum, and evaluates either `spectrum_chisq`
or a user-supplied loss function.
"""
struct SpectrumFitProblem{E,P,D,L}
    evaluator::E
    initial_params::P
    data::D
    loss::L
    parameter_names::Vector{String}
end

function _validate_fit_dataset(data::SpectrumDataset, project::Project)
    for group in data.groups
        group.config <= length(project.configs) || throw(ArgumentError(
            "dataset config index $(group.config) exceeds the Project's " *
            "$(length(project.configs)) configs"))
        cfg = project.configs[group.config]
        irrep_index = findfirst(==(group.irrep), cfg.irreps)
        irrep_index === nothing && throw(ArgumentError(
            "config $(group.config) does not request irrep " *
            repr(group.irrep)))
        requested = cfg.n_levels[irrep_index]
        if group.levels === nothing
            length(group.values) == requested || throw(DimensionMismatch(
                "config $(group.config), irrep $(group.irrep) contains " *
                "$(length(group.values)) data values but Project requests " *
                "$requested levels; provide explicit levels to fit a subset"))
        else
            maximum(group.levels) <= requested || throw(DimensionMismatch(
                "config $(group.config), irrep $(group.irrep) requests only " *
                "$requested levels, but the dataset selects level " *
                "$(maximum(group.levels))"))
        end
    end
    return data
end

function _fit_parameter_metadata(initial_params)
    P = typeof(initial_params)
    names = try
        String.(param_names(P))
    catch err
        throw(ArgumentError(
            "initial_params must be an instance of a type created by @params; " *
            "param_names($P) is unavailable: $(sprint(showerror, err))"))
    end
    isempty(names) && throw(ArgumentError(
        "a spectrum fit requires at least one parameter"))
    return names
end

function SpectrumFitProblem(prepared::PreparedSpectrumProject, V_func,
                            initial_params, data::SpectrumDataset)
    _validate_fit_dataset(data, prepared.project)
    names = _fit_parameter_metadata(initial_params)
    evaluator = params -> compute!(prepared, V_func, params)
    loss = (result, params) -> spectrum_chisq(result, data)
    return SpectrumFitProblem(evaluator, initial_params, data, loss, names)
end

function SpectrumFitProblem(prepared::AffinePreparedProject{P},
                            initial_params::P,
                            data::SpectrumDataset) where P
    _validate_fit_dataset(data, prepared.project)
    names = _fit_parameter_metadata(initial_params)
    names == prepared.parameter_names || throw(ArgumentError(
        "initial parameter names do not match affine preparation"))
    evaluator = params -> compute!(prepared, params)
    loss = (result, params) -> spectrum_chisq(result, data)
    return SpectrumFitProblem(evaluator, initial_params, data, loss, names)
end

function (problem::SpectrumFitProblem)(x::AbstractVector)
    length(x) == length(problem.parameter_names) || throw(DimensionMismatch(
        "fit vector has length $(length(x)); expected " *
        "$(length(problem.parameter_names))"))
    all(isfinite, x) || throw(ArgumentError(
        "fit parameter vector must contain only finite values"))
    params = from_vector(typeof(problem.initial_params), x)
    result = problem.evaluator(params)
    value = problem.loss(result, params)
    value isa Real || throw(ArgumentError(
        "fit loss must return a real scalar, got $(typeof(value))"))
    isfinite(value) || throw(ArgumentError(
        "fit loss must return a finite value, got $value"))
    return Float64(value)
end

# ============================================================
# Interactive mode
# ============================================================

"""
    setup_project() -> Project

Interactively construct a `Project` in the REPL in two stages:

Stage A: enter total momentum `d`, total isospin `I`, Fock channels, and their `Ncut` values.
Stage B: repeatedly add `(L,a,Γ,n_Γ)` configurations; enter an empty `L` to finish.

Returns a `Project` containing all configurations, ready for `compute!`.

# Example
```julia
proj = setup_project()
result = compute!(proj, my_V, my_params)
result[1][\"T1-\"]   # T1- energy levels of the first configuration
```
"""
function setup_project()
    println("="^56)
    println("  Interactive NPHFforFVE Project setup")
    println("="^56)

    # ===== Stage A: d, I, channels, Ncuts =====
    println("\n── Stage A: define the physical system ──")

    print("Total momentum d (format: nx ny nz; default: 0 0 0): ")
    d_input = strip(readline())
    d = if isempty(d_input)
        D000
    else
        parts = parse.(Int, split(d_input))
        length(parts) == 3 || throw(ArgumentError("total momentum must contain three integers"))
        Momentum(parts...)
    end
    println("  d = $d")

    print("Total isospin I (for example 0, 1/2, 1; default: 0): ")
    I_input = strip(readline())
    I = isempty(I_input) ? 0//1 : _parse_rational(I_input)
    println("  I = $I")

    print("\nNumber of Fock channels: ")
    n_ch = parse(Int, readline())
    n_ch >= 1 || throw(ArgumentError("at least one channel is required"))

    channels = FockChannel[]
    Ncuts = Int[]

    for i in 1:n_ch
        println("\n--- Channel $i ---")
        ch = _prompt_fock_channel(i)
        push!(channels, ch)
        if ch.N > 1
            print("  Ncut for this channel (|n|² ≤ Ncut): ")
            nc = parse(Int, readline())
            nc >= 1 || throw(ArgumentError("Ncut must be ≥ 1 for a channel with N > 1"))
            push!(Ncuts, nc)
        else
            println("  For an N=1 channel, Ncut is automatically set to 0 (its momentum is fixed by d)")
            push!(Ncuts, 0)
        end
    end

    proj = Project(d, I, channels, Ncuts)
    println("\n✓ Stage A complete: $(n_ch) channels")

    # ===== Stage B: configurations =====
    println("\n── Stage B: add configurations (L,a,Γ,n_Γ) ──")
    println("Available irreps: $(join(OH_IRREP_NAMES, ", "))")
    println("Enter an empty L to finish adding configurations\n")

    while true
        print("L (lattice extent; press Enter to finish): ")
        L_input = strip(readline())
        isempty(L_input) && break
        L = parse(Int, L_input)

        print("a (lattice spacing in fm): ")
        a = parse(Float64, readline())

        print("Requested irreps (space-separated): ")
        irr_input = strip(readline())
        isempty(irr_input) && throw(ArgumentError("at least one irrep is required"))
        irreps = String[String(s) for s in split(irr_input)]

        print("Numbers of energy levels (space-separated; $(length(irreps)) values): ")
        n_input = strip(readline())
        n_levels = parse.(Int, split(n_input))

        idx = add_config!(proj, L, a, irreps, n_levels)
        println("  ✓ Added configuration #$idx ($(length(irreps)) irrep(s))\n")
    end

    isempty(proj.configs) && @warn("no configurations were added; compute! will throw an error")

    # ===== Summary =====
    println("="^56)
    println("Project setup complete")
    println("  Total momentum d = $(proj.d)")
    println("  Total isospin I = $(proj.I)")
    println("  Number of channels: $(length(proj.channels))")
    for (i, ch) in enumerate(proj.channels)
        println("    Channel $i: \"$(ch.name)\" N=$(ch.N) Ncut=$(proj.Ncuts[i])")
    end
    println("  Number of configurations: $(length(proj.configs))")
    for (i, cfg) in enumerate(proj.configs)
        println("    #$i: L=$(cfg.L), a=$(cfg.a), Γ=$(join(cfg.irreps, ","))")
    end
    println("="^56)
    println("\nNext step: result = compute!(proj, V_func, params)")
    println("        result[idx][\"Γ\"] retrieves energy levels")

    println()
    path = generate_potential_template(proj)
    println("✓ Template generated: $path")
    println("Fill in the my_V(...) matrix elements and MyParams parameters in the template")

    return proj
end

# ============ Interactive helpers ============

function _prompt_fock_channel(i::Int)
    print("  Channel name (for example \"rho\"): ")
    name = strip(readline())
    isempty(name) && throw(ArgumentError("channel name must not be empty"))

    print("  Number of particle species: ")
    n_species = parse(Int, readline())
    n_species >= 1 || throw(ArgumentError("the number of particle species must be ≥ 1"))

    species = Int[]
    ptypes  = Symbol[]
    masses  = Float64[]
    spins   = Rational{Int}[]
    isospins = Rational{Int}[]
    etas    = Float64[]

    for s in 1:n_species
        println("  Particle species $s:")
        print("    Number of particles: ")
        push!(species, parse(Int, readline()))
        print("    Type (boson/fermion): ")
        pt = Symbol(lowercase(strip(readline())))
        pt in (:boson, :fermion) || throw(ArgumentError("particle type must be boson or fermion"))
        push!(ptypes, pt)
        print("    Mass (MeV): ")
        push!(masses, parse(Float64, readline()))
        print("    Spin j (0, 1/2, 1): ")
        push!(spins, _parse_rational(readline()))
        print("    Isospin i (0, 1/2, 1): ")
        push!(isospins, _parse_rational(readline()))
        print("    Intrinsic parity η (+1/-1): ")
        push!(etas, parse(Float64, readline()))
    end

    print("  Kinetic-energy dispersion (relativistic/nonrelativistic; default: relativistic): ")
    kt_input = strip(readline())
    kt = if isempty(kt_input) || lowercase(kt_input) == "relativistic"
        relativistic
    elseif lowercase(kt_input) == "nonrelativistic"
        nonrelativistic
    else
        throw(ArgumentError("dispersion must be relativistic or nonrelativistic"))
    end

    return FockChannel(name, species, ptypes, masses, spins, isospins, etas, kt)
end

function _parse_rational(s::AbstractString)
    s = strip(s)
    if occursin("//", s)
        parts = split(s, "//")
        length(parts) == 2 || throw(ArgumentError("cannot parse rational number: $s"))
        return parse(Int, parts[1]) // parse(Int, parts[2])
    elseif occursin('/', s)
        parts = split(s, '/')
        length(parts) == 2 || throw(ArgumentError("cannot parse rational number: $s"))
        return parse(Int, parts[1]) // parse(Int, parts[2])
    else
        return parse(Int, s) // 1
    end
end

# ============================================================
# Interaction-template generation
# ============================================================

"""
    generate_potential_template(proj::Project, output_file::String="potential_defs.jl") -> String

Generate an interaction-function template from the Fock channels and total isospin in `proj`.

- `output_file`: output filename; defaults to `"potential_defs.jl"`

The template contains:
- an `@params` parameter-structure skeleton;
- a `my_V_αβ` function skeleton for every channel pair, including isospin-subchannel branches.

Only the physical matrix elements need to be filled in.

Returns the generated file path.
"""
function generate_potential_template(proj::Project, output_file::String="potential_defs.jl")
    V_basis = proj.V_basis
    path = output_file
    io = open(path, "w")

    I = proj.I
    n_ch = length(proj.channels)

    # ===== File header =====
    println(io, "# ============================================================")
    println(io, "# Interaction Potential V — User-Defined Matrix Elements")
    println(io, "# ============================================================")
    println(io, "#")
    println(io, "# Project Summary:")
    println(io, "#   Total momentum: d = $(proj.d)")
    println(io, "#   Total isospin: I = $(_math_fraction(proj.I))")
    for (i, ch) in enumerate(proj.channels)
        println(io, "#   ch $i: \"$(ch.name)\" N=$(ch.N) species=$(ch.species)")
    end
    for (i, cfg) in enumerate(proj.configs)
        println(io, "#   config $i: L=$(cfg.L), a=$(cfg.a), Γ=$(join(cfg.irreps, ","))")
    end
    if !isempty(proj.exclude_subchannels)
        println(io, "#   Excluded isospin subchannels:")
        for ex in proj.exclude_subchannels
            r_label = ex.r === nothing ? "all r" : "r=$(ex.r)"
            println(io, "#     $(ex.channel_name): κ=$(_math_kappa(ex.kappa)), $r_label")
        end
        println(io, "#   Physical r labels retain their original values and may be nonconsecutive.")
    end
    println(io, "#")
    println(io, "# Momentum convention: p = (2πħc/L_phys)n, with ħc = 197.327 MeV·fm.")
    println(io, "# pA[i] and pB[i] are precomputed SVector{3,Float64} momenta in MeV.")
    println(io, "# ============================================================")
    println(io)
    println(io, "using NPHFforFVE")
    println(io)

    # ===== _PER_SPIN (helicity basis only) =====
    if V_basis == :helicity
        println(io, "# ============ Per-particle spins (for ZM detection) ============")
        println(io, "# _PER_SPIN[ch] = spin of each particle in channel ch")
        println(io, "# Used to detect spinful particles at zero momentum (zmA, zmB).")
        println(io, "const _PER_SPIN = [")
        for (i, ch) in enumerate(proj.channels)
            per_spin = Float64[]
            for (s, j) in zip(ch.species, ch.spins)
                append!(per_spin, fill(Float64(j), s))
            end
            println(io, "    $(per_spin),  # ch $i: \"$(ch.name)\"")
        end
        println(io, "]")
        println(io)
    end

    # ===== @params =====
    println(io, "# ============ Parameters ============")
    println(io, "# Define LECs, cutoffs, and other model parameters below.")
    println(io, "@params struct MyParams")
    println(io, "    # Examples (uncomment and edit):")
    println(io, "    # C0  = 1.0    # leading-order contact")
    println(io, "    # C1  = 0.5    # NLO contact")
    println(io, "    # Λ   = 1000.0 # cutoff (MeV)")
    println(io, "end")
    println(io)

    # ===== Unified V function =====
    println(io, "# ============ Interaction Matrix Elements ============")
    println(io, "#")
    println(io, "# Signature:")
    if V_basis == :canonical
        println(io, "#   my_V(nA, nB, sp, s, kapA, kapB, rA, rB, chA, chB, L_phys, params)")
        println(io, "#")
        println(io, "#   nA, nB       : bra and ket momenta; NTuple{N,Momentum}")
        println(io, "#   sp, s        : bra and ket spin projections; NTuple{N,Rational{Int}}")
        println(io, "#   kapA, kapB   : κ labels; String, or Tuple for multiple species")
        println(io, "#   rA, rB       : physical subchannel indices at fixed κ")
        println(io, "#   chA, chB     : channel indices listed below")
        println(io, "#   L_phys       : spatial extent in fm")
        println(io, "#   params       : MyParams instance")
        println(io, "#")
        println(io, "# Return the dim(κA) × dim(κB) interaction block; a scalar is allowed when both dimensions are 1.")
        println(io, "#")
        println(io, "# Hermiticity: my_V(nA,nB,...,chA,chB,...) = conj(my_V(nB,nA,...,chB,chA,...)).")
    else
        println(io, "#   my_V_hel(nA, nB, lamA, lamB, kapA, kapB, rA, rB, chA, chB, L_phys, params)")
        println(io, "#")
        println(io, "#   nA, nB : bra / ket momentum tuple  NTuple{N, Momentum}")
        println(io, "#   lamA, lamB : bra / ket helicity  NTuple{N, Float64}")
        println(io, "#        (degenerates to canonical spin projection σ for ZM+spin particles)")
        println(io, "#   kapA, kapB : S_N irrep label (String, or Tuple for multi-species)")
        println(io, "#   rA, rB : physical sub-channel index at fixed kap (see map below)")
        println(io, "#   chA, chB : channel index (see table below)")
        println(io, "#   params : MyParams instance")
        println(io, "#")
        println(io, "#   Return a dimA × dimB matrix (scalar if dimA=dimB=1) with")
        println(io, "#   the V matrix elements between the given isospin sub-channels.")
        println(io, "#")
        println(io, "# IMPORTANT: V must be Hermitian: my_V_hel(nA, nB, ..., chA, chB, L_phys, params)")
        println(io, "#            = conj(my_V_hel(nB, nA, ..., chB, chA, L_phys, params))")
    end
    println(io)

    # Channel index table
    _write_channel_table(io, proj, I)
    println(io)

    # Detect a moving frame
    is_moving = proj.d != D000

    # Generate the unified function dispatched by (chA, chB).
    fname = V_basis == :helicity ? "my_V_hel" : "my_V"
    spin_args = V_basis == :helicity ? "lamA, lamB" : "sp, s"
    println(io, "function $fname(nA, nB, $spin_args, kapA, kapB, rA, rB, chA, chB, L_phys, params)")
    println(io, "    # ===== Momentum Conversion =====")
    println(io, "    # p = (2πħc/L_phys)n, with ħc = 197.327 MeV·fm.")
    println(io, "    pv = 2π * 197.327 / L_phys")

    # ZM detection for helicity basis (only if any channel has spinful particles)
    _has_any_spin = any(any(j -> j != 0, ch.spins) for ch in proj.channels)
    if V_basis == :helicity && _has_any_spin
        println(io)
        println(io, "    # ── Zero-momentum detection ──")
        println(io, "    # Spinful particles at rest: helicity λ degenerates to canonical spin σ.")
        println(io, "    # zmA[i]/zmB[i] = true  ⇒  lamA[i]/lamB[i] ∈ {j, j-1, ..., -j}")
        println(io, "    zmA = [iszero(nA[i]) && _PER_SPIN[chA][i] != 0.0 for i in 1:length(nA)]")
        println(io, "    zmB = [iszero(nB[i]) && _PER_SPIN[chB][i] != 0.0 for i in 1:length(nB)]")
    end

    if is_moving
        # Moving frame: per-particle expanded masses plus boost.
        println(io)
        println(io, "    # ── Lorentz boost to CM frame (moving system) ──")
        println(io, "    # Per-channel per-particle masses:")
        println(io, "    _masses_by_ch = [")
        for (i, ch) in enumerate(proj.channels)
            expanded = Any[]
            for (count, mass) in zip(ch.species, ch.masses)
                append!(expanded, fill(mass, count))
            end
            if any(m -> m isa DynamicMass, expanded)
                specs = "[" * join(repr.(expanded), ", ") * "]"
                println(io, "        [resolve_mass(m, params) for m in $specs]," *
                            "  # ch $(i): $(ch.name)")
            else
                println(io, "        $(Float64.(expanded)),  # ch $(i): $(ch.name)")
            end
        end
        println(io, "    ]")
        println(io, "    _d_tot = $(proj.d)")
        println(io)
        println(io, "    pA_mov = [pv .* Float64.(n) for n in nA]")
        println(io, "    pB_mov = [pv .* Float64.(n) for n in nB]")
        println(io, "    pA, facA = boost_to_cm(pA_mov, _masses_by_ch[chA], _d_tot, L_phys)")
        println(io, "    pB, facB = boost_to_cm(pB_mov, _masses_by_ch[chB], _d_tot, L_phys)")
    else
        println(io, "    # pA[i] and pB[i] are SVector{3,Float64} momenta in MeV.")
        println(io, "    pA = [pv .* Float64.(n) for n in nA]")
        println(io, "    pB = [pv .* Float64.(n) for n in nB]")
        println(io, "    facA = 1.0; facB = 1.0")
    end
    println(io)

    # Return-value wrapper: moving frames require a kinematic factor.
    ret_wrap = is_moving ? ("facA * (", ") * facB") : ("", "")

    first_block = true
    emitted_hermitian = Set{Tuple{Int,Int}}()  # mark (chA,chB) pairs handled by a Hermitian-conjugate comment

    for α in 1:n_ch, β in 1:n_ch
        (α, β) in emitted_hermitian && continue
        ch_α = proj.channels[α]
        ch_β = proj.channels[β]
        subs_α = _active_subchannels(ch_α, I, proj.exclude_subchannels)
        subs_β = _active_subchannels(ch_β, I, proj.exclude_subchannels)
        multi_α = length(ch_α.species) > 1
        multi_β = length(ch_β.species) > 1

        keyword = first_block ? "if" : "elseif"
        arrow = "←"

        if α == β
            # Diagonal block
            println(io, "    # ── $(ch_α.name) ($(ch_α.N)-body) diagonal ──")
            println(io, "    $keyword chA == $α && chB == $β")
            _write_channel_pair_body(io, ch_α, ch_β, subs_α, subs_β,
                                     multi_α, multi_β, α, β, arrow;
                                     ret_prefix=ret_wrap[1], ret_suffix=ret_wrap[2],
                                     V_basis=V_basis)
        else
            # α<β: write chA←chB; label chB←chA as its Hermitian conjugate
            other = (β, α)
            push!(emitted_hermitian, other)

            println(io, "    # ── $(ch_α.name) ← $(ch_β.name) ──")
            println(io, "    $keyword chA == $α && chB == $β")
            _write_channel_pair_body(io, ch_α, ch_β, subs_α, subs_β,
                                     multi_α, multi_β, α, β, arrow;
                                     ret_prefix=ret_wrap[1], ret_suffix=ret_wrap[2],
                                     V_basis=V_basis)

            println(io)
            keyword = "elseif"
            println(io, "    # ── $(ch_β.name) ← $(ch_α.name)  (Hermitian conjugate of $α←$β) ──")
            println(io, "    $keyword chA == $β && chB == $α")
            _write_hermitian_conj_body(io, α, β; V_basis=V_basis)
        end

        first_block = false
    end

    println(io, "    end")
    println(io, "    error(\"unreachable: no matching channel pair for chA=\$chA chB=\$chB\")")
    println(io, "end")
    println(io)

    # ===== Sparse filters (optional) =====
    println(io, "# ============ Optional Sparse Filters ============")
    println(io, "# Return false only when the selected object is known to be exactly zero.")
    println(io, "#")
    println(io, "# Skip a complete Fock-channel pair.")
    println(io, "function channel_filter(chA, chB, params)")
    println(io, "    # chA == 1 && chB == 3 && return false")
    println(io, "    return true")
    println(io, "end")
    println(io)
    println(io, "# Skip one specific interaction matrix element.")
    println(io, "function entry_filter(nA, nB, $spin_args, kapA, kapB, rA, rB, chA, chB, L_phys, params)")
    println(io, "    # nA[2] != nB[2] && return false  # spectator matching")
    println(io, "    return true")
    println(io, "end")
    println(io)

    close(io)
    return abspath(path)
end

# ===== Channel index table =====

_math_fraction(x::Rational) = denominator(x) == 1 ? string(numerator(x)) :
                              "$(numerator(x))/$(denominator(x))"
_math_fraction(x) = string(x)

function _math_tuple(xs; formatter=_math_fraction)
    values = join((formatter(x) for x in xs), ", ")
    return length(xs) == 1 ? "($values,)" : "($values)"
end

_math_kappa(κ::AbstractString) = κ
_math_kappa(κ::Tuple) = join(κ, " ⊗ ")

const _SUBSCRIPT_DIGITS = ("₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉")
function _subscript(n::Integer)
    return join(_SUBSCRIPT_DIGITS[parse(Int, d) + 1] for d in string(n))
end

function _species_isospin_label(Js::Tuple)
    return join(("I$(_subscript(i)) = $(_math_fraction(J))"
                 for (i, J) in enumerate(Js)), ", ")
end

function _coupling_path_label(path::Tuple)
    isempty(path) && return "—"
    labels = String[]
    for (i, J) in enumerate(path)
        indices = join(_subscript(k) for k in 1:(i + 1))
        push!(labels, "I$indices = $(_math_fraction(J))")
    end
    return join(labels, ", ")
end

function _square_root_integer(n::Integer)
    root = isqrt(n)
    return root * root == n ? root : nothing
end

function _charge_coefficient_magnitude(value::Real)
    magnitude = abs(Float64(value))
    isapprox(magnitude, 1.0; atol=1e-11) && return ""
    squared = rationalize(Int, magnitude^2; tol=1e-10)
    p, q = numerator(squared), denominator(squared)
    p_root, q_root = _square_root_integer(p), _square_root_integer(q)
    if p_root !== nothing && q_root !== nothing
        return q_root == 1 ? string(p_root) : "$(p_root)/$(q_root)"
    elseif p_root !== nothing
        return p_root == 1 ? "1/√$(q)" : "$(p_root)/√$(q)"
    elseif q_root !== nothing
        return q_root == 1 ? "√$(p)" : "√$(p)/$(q_root)"
    elseif p == 1
        return "1/√$(q)"
    elseif q == 1
        return "√$(p)"
    end
    return "√($(p)/$(q))"
end

function _charge_ket(state::Tuple, species::Vector{Int})
    groups = String[]
    offset = 0
    for count in species
        entries = join((_math_fraction(state[offset + i]) for i in 1:count), ",")
        push!(groups, entries)
        offset += count
    end
    return "|" * join(groups, "; ") * "⟩"
end

function _charge_expansion(coefficients::Dict, species::Vector{Int})
    terms = String[]
    states = sort(collect(keys(coefficients)); rev=true)
    for state in states
        coefficient = coefficients[state]
        abs(imag(coefficient)) < 1e-11 ||
            throw(ArgumentError("charge-basis coefficient is unexpectedly complex"))
        negative = real(coefficient) < 0
        magnitude = _charge_coefficient_magnitude(real(coefficient))
        term = isempty(magnitude) ? _charge_ket(state, species) :
                                   magnitude * " " * _charge_ket(state, species)
        if isempty(terms)
            push!(terms, negative ? "−" * term : term)
        else
            push!(terms, (negative ? " − " : " + ") * term)
        end
    end
    return isempty(terms) ? "0" : join(terms)
end

function _write_charge_basis_expansions(io, ch, I, subs)
    println(io, "#")
    println(io, "#     Charge-State Expansions at M = I = $(_math_fraction(I))")
    println(io, "#     For reference, the following equations give one transformation from the isospin basis to charge states.")
    length(ch.species) > 1 &&
        println(io, "#     Semicolons in each ket separate the species listed above.")
    for sub in subs, a in 1:sub.dim
        coefficients = _subchannel_charge_coefficients(
            ch.species, ch.isospins, I, sub, a; M=I)
        a_label = sub.dim == 1 ? "" : ", a=$a"
        lhs = "|I=$(_math_fraction(I)), M=$(_math_fraction(I)); " *
              "κ=$(_math_kappa(sub.κ)), r=$(sub.r)$(a_label)⟩"
        println(io, "#     $lhs = $(_charge_expansion(coefficients, ch.species))")
    end
end

_mass_display(mass::DynamicMass) = "params.$(mass.name)"
_mass_display(mass::Real) = string(mass)

function _write_channel_table(io, proj, I)
    println(io, "# | Channel | Name | N | Species | κ at I = $(_math_fraction(I)) |")
    println(io, "# |---:|:---|---:|:---|:---|")
    for (i, ch) in enumerate(proj.channels)
        subs = _active_subchannels(ch, I, proj.exclude_subchannels)
        kappas = join((_math_kappa(κ) for κ in unique(s.κ for s in subs)), ", ")
        println(io, "# | $i | $(ch.name) | $(ch.N) | $(ch.species) | $kappas |")
    end
    println(io)
    println(io, "# Channel Details")
    for (i, ch) in enumerate(proj.channels)
        kt_str = ch.kinetic_type == nonrelativistic ? "nonrelativistic" : "relativistic"
        println(io, "#   ch $i: \"$(ch.name)\" ($(ch.N)-body, $kt_str)")
        for s in 1:length(ch.species)
            pt = ch.particle_types[s] == :boson ? "boson" : "fermion"
            println(io, "#     sp.$s: N=$(ch.species[s]), $pt, m=$(_mass_display(ch.masses[s])) MeV, " *
                        "j=$(_math_fraction(ch.spins[s])), I=$(_math_fraction(ch.isospins[s])), η=$(ch.etas[s])")
        end
        subs = _active_subchannels(ch, I, proj.exclude_subchannels)
        println(io, "#     Isospin subchannels (Iₛ follows the species order above).")
        println(io, "#     Intermediate isospins use left-associated coupling; — denotes direct coupling.")
        if isempty(subs)
            println(io, "#     No subchannels at total I=$(_math_fraction(I)).")
        else
            show_coupling_path = any(!isempty(sub.coupling_path) for sub in subs)
            show_internal_copy = any(any(μ -> μ != 1, sub.multiplicity_tuple) for sub in subs)
            println(io, "#")
            if show_coupling_path && show_internal_copy
                println(io, "#     | κ | r | Subsystem isospins | Intermediate isospins | Internal copy | dim(κ) |")
                println(io, "#     |:---|---:|:---|:---|:---|---:|")
            elseif show_coupling_path
                println(io, "#     | κ | r | Subsystem isospins | Intermediate isospins | dim(κ) |")
                println(io, "#     |:---|---:|:---|:---|---:|")
            elseif show_internal_copy
                println(io, "#     | κ | r | Subsystem isospins | Internal copy | dim(κ) |")
                println(io, "#     |:---|---:|:---|:---|---:|")
            else
                println(io, "#     | κ | r | Subsystem isospins | dim(κ) |")
                println(io, "#     |:---|---:|:---|---:|")
            end
            for sub in subs
                row = "#     | $(_math_kappa(sub.κ)) | $(sub.r) | " *
                      "$(_species_isospin_label(sub.J_tuple)) | "
                if show_coupling_path
                    row *= "$(_coupling_path_label(sub.coupling_path)) | "
                end
                if show_internal_copy
                    row *= "μ = $(_math_tuple(sub.multiplicity_tuple; formatter=string)) | "
                end
                println(io, row * "$(sub.dim) |")
            end

            composite_kappas = unique(sub.κ for sub in subs
                                      if sub.κ isa Tuple && sub.dim > 1)
            for κ in composite_kappas
                println(io, "#")
                println(io, "#     Carrier-Space Order for κ = $(_math_kappa(κ))")
                println(io, "#     X and both indices of V use this order.")
                labels = join(("a$(_subscript(s))" for s in eachindex(ch.species)), ", ")
                println(io, "#")
                println(io, "#     | Matrix index a | Carrier indices ($labels) |")
                println(io, "#     |---:|:---|")
                for (a, indices) in enumerate(_carrier_index_tuples(ch.species, κ))
                    assignments = join(("a$(_subscript(s)) = $(indices[s])"
                                        for s in eachindex(indices)), ", ")
                    println(io, "#     | $a | $assignments |")
                end
            end
            _write_charge_basis_expansions(io, ch, I, subs)
        end
    end
end

"""
    generate_subchannel_report(proj, output_file="subchannels.md"; overwrite=false)

Write a Markdown report of every physical isospin subchannel at the Project's
total isospin. Excluded entries remain visible and are marked as excluded.
"""
function generate_subchannel_report(proj::Project,
                                    output_file::AbstractString="subchannels.md";
                                    overwrite::Bool=false)
    path = String(output_file)
    isfile(path) && !overwrite && throw(ArgumentError(
        "target file already exists: $(abspath(path)); pass overwrite=true to replace it"))
    all_subs = [(ch, get_isospin_subchannels(ch, proj.I)) for ch in proj.channels]
    any(!isempty(subs) for (_, subs) in all_subs) || throw(ArgumentError(
        "Project has no isospin subchannels at total isospin I=$(proj.I)"))

    open(path, "w") do io
        println(io, "# Isospin-subchannel report")
        println(io)
        println(io, "- Total momentum: `$(proj.d)`")
        println(io, "- Total isospin: \$I=$(_math_fraction(proj.I))\$")
        println(io, "- Status: the report lists the full physical space; entries marked as excluded do not enter the final template or numerical basis.")
        println(io, "- Numbering: \$r\$ is a stable physical-subchannel index and is not renumbered after exclusions.")
        println(io)
        println(io, "## Fock-channel overview")
        println(io)
        println(io, "| Index | Name | Particle count | Particles per species |")
        println(io, "|---:|:---|---:|:---|")
        for (i, ch) in enumerate(proj.channels)
            println(io, "| $i | $(ch.name) | $(ch.N) | `$(ch.species)` |")
        end

        for (channel_index, (ch, subs)) in enumerate(all_subs)
            println(io)
            println(io, "## $channel_index. $(ch.name)")
            println(io)
            for s in eachindex(ch.species)
                stat = ch.particle_types[s] == :boson ? "boson" : "fermion"
                println(io, "- Species $(s): $(ch.species[s]) × $(stat), " *
                            "\$m=$(_mass_display(ch.masses[s]))\$ MeV, \$j=$(_math_fraction(ch.spins[s]))\$，" *
                            "\$I=$(_math_fraction(ch.isospins[s]))\$, \$\\eta=$(ch.etas[s])\$")
            end
            println(io)
            if isempty(subs)
                println(io, "This Fock channel has no subchannels at the specified total isospin.")
                continue
            end

            show_path = any(!isempty(sub.coupling_path) for sub in subs)
            show_copy = any(any(!=(1), sub.multiplicity_tuple) for sub in subs)
            headers = [raw"$\kappa$", raw"$r$", "Subsystem isospin"]
            show_path && push!(headers, "Intermediate isospin")
            show_copy && push!(headers, "Internal copy")
            append!(headers, [raw"$\dim(\kappa)$", "Status"])
            println(io, "| " * join(headers, " | ") * " |")
            println(io, "|" * join(fill(":---", length(headers)), "|") * "|")
            for sub in subs
                excluded = _is_subchannel_excluded(
                    proj.exclude_subchannels, ch.name, sub)
                values = ["$(_math_kappa(sub.κ))", string(sub.r),
                          _species_isospin_label(sub.J_tuple)]
                show_path && push!(values, _coupling_path_label(sub.coupling_path))
                show_copy && push!(values,
                    "μ = $(_math_tuple(sub.multiplicity_tuple; formatter=string))")
                append!(values, [string(sub.dim), excluded ? "Excluded" : "Included"])
                println(io, "| " * join(values, " | ") * " |")
            end

            composite_kappas = unique(sub.κ for sub in subs
                                      if sub.κ isa Tuple && sub.dim > 1)
            for kappa in composite_kappas
                println(io)
                println(io, "### Carrier-space ordering: $(_math_kappa(kappa))")
                println(io)
                println(io, "This ordering is used for both the projection matrix \$X\$ and interaction-matrix indices.")
                println(io)
                println(io, "| Matrix index \$a\$ | Composite carrier indices |")
                println(io, "|---:|:---|")
                for (a, indices) in enumerate(_carrier_index_tuples(ch.species, kappa))
                    assignments = join(("a$(_subscript(s)) = $(indices[s])"
                                        for s in eachindex(indices)), ", ")
                    println(io, "| $a | $assignments |")
                end
            end

            println(io)
            println(io, "### Charge-state expansion at \$I_z=I\$")
            println(io)
            println(io, "The following gives one linear combination that expresses the isospin basis in charge states, for reference.")
            length(ch.species) > 1 &&
                println(io, "In a ket, semicolons separate species in the order listed above.")
            println(io)
            for sub in subs, a in 1:sub.dim
                coefficients = _subchannel_charge_coefficients(
                    ch.species, ch.isospins, proj.I, sub, a; M=proj.I)
                a_label = sub.dim == 1 ? "" : ", a=$a"
                lhs = "|I=$(_math_fraction(proj.I)), I_z=$(_math_fraction(proj.I)); " *
                      "κ=$(_math_kappa(sub.κ)), r=$(sub.r)$(a_label)⟩"
                println(io, raw"$$")
                println(io, lhs * " = " * _charge_expansion(coefficients, ch.species))
                println(io, raw"$$")
            end

            println(io)
            println(io, "### Exclusion commands")
            println(io)
            println(io, "Place the following commands before generating the final interaction template:")
            println(io)
            println(io, "```julia")
            for sub in subs
                println(io, "# exclude_subchannel!(project, $(repr(ch.name)), " *
                            "$(repr(sub.κ)), $(sub.r))")
            end
            for kappa in unique(sub.κ for sub in subs)
                println(io, "# exclude_subchannel!(project, $(repr(ch.name)), " *
                            "$(repr(kappa)))  # exclude all r under this κ")
            end
            println(io, "```")
        end
    end
    return abspath(path)
end

# ===== Channel-pair branch body =====

function _write_channel_pair_body(io, ch_α, ch_β, subs_α, subs_β,
                                  multi_α, multi_β, α, β, arrow;
                                  ret_prefix="", ret_suffix="",
                                  V_basis::Symbol=:canonical)
    # Empty-subchannel guard: this channel has no isospin subchannels at the given I.
    if isempty(subs_α) || isempty(subs_β)
        name = α == β ? ch_α.name : "$(ch_α.name)←$(ch_β.name)"
        println(io, "        # $name: no isospin sub-channels at this I — V ≡ 0")
        println(io, "        return $(ret_prefix)0.0$(ret_suffix)")
        return
    end

    if V_basis == :helicity
        # ZM branching only needed if at least one channel has spinful particles
        _pair_has_spin = any(j -> j != 0, ch_α.spins) || any(j -> j != 0, ch_β.spins)
        if _pair_has_spin
            println(io, "        if any(zmA) || any(zmB)")
            println(io, "            # ═══ ZM BRANCH ═══")
            println(io, "            # Spinful ZM particles: lamA[i]/lamB[i] = canonical σ_i")
            println(io, "            # Use lamA[i], lamB[i] directly as spin projection values.")
            _write_full_branches_body(io, subs_α, subs_β, ch_α.name, ch_β.name;
                                      ret_prefix=ret_prefix, ret_suffix=ret_suffix,
                                      indent="            ", zm_mode="ZM")
            println(io, "        else")
            println(io, "            # ═══ HELICITY BRANCH ═══")
            println(io, "            # All momenta non-zero: standard helicity formula.")
            _write_full_branches_body(io, subs_α, subs_β, ch_α.name, ch_β.name;
                                      ret_prefix=ret_prefix, ret_suffix=ret_suffix,
                                      indent="            ", zm_mode="helicity")
            println(io, "        end")
        else
            # all-scalar pair: no ZM possible, helicity branch directly
            _write_full_branches_body(io, subs_α, subs_β, ch_α.name, ch_β.name;
                                      ret_prefix=ret_prefix, ret_suffix=ret_suffix,
                                      indent="        ", zm_mode=nothing)
        end
    else
        _write_full_branches_body(io, subs_α, subs_β, ch_α.name, ch_β.name;
                                  ret_prefix=ret_prefix, ret_suffix=ret_suffix)
    end
end

function _write_hermitian_conj_body(io, α, β; V_basis::Symbol=:canonical)
    fname = V_basis == :helicity ? "my_V_hel" : "my_V"
    # For Hermitian conjugate: bra↔ket, so old lamB (ket) → new lamA (bra), old lamA (bra) → new lamB (ket)
    spin_args = V_basis == :helicity ? "lamB, lamA" : "s, sp"
    println(io, "        # Hermitian conjugate: V(chB,chA) = conj(V(chA,chB))")
    println(io, "        return conj($fname(nB, nA, $spin_args, kapB, kapA, rB, rA, chB, chA, L_phys, params))")
end

# ===== General branches (enumerate all κ, r, a combinations; do not enforce Wigner–Eckart). =====

function _write_full_branches_body(io, subs_α, subs_β, name_α, name_β;
                                   ret_prefix="", ret_suffix="",
                                   indent="        ",
                                   zm_mode::Union{Nothing, String}=nothing)
    need_κ = _need_branch_value(subs_α, :κ) || _need_branch_value(subs_β, :κ)
    need_r = _need_branch_value(subs_α, :r) || _need_branch_value(subs_β, :r)
    any_branch = need_κ || need_r
    indent2 = indent * "    "  # 4 more spaces for body

    if !any_branch
        s_α = subs_α[1]
        s_β = subs_β[1]
        tag = zm_mode === nothing ? "" : " ($zm_mode)"
        println(io, "$indent# ⟨κA=$(_math_kappa(s_α.κ)), rA=$(s_α.r) | V | κB=$(_math_kappa(s_β.κ)), rB=$(s_β.r)⟩$tag")
        println(io, "$(indent)error(\"TODO: fill matrix block for κA=$(_escape_str(s_α.κ)), κB=$(_escape_str(s_β.κ)), rA=$(s_α.r), rB=$(s_β.r)\")")
        return
    end

    first = true
    for s_α in subs_α, s_β in subs_β
        conds = String[]
        need_κ && push!(conds, "kapA == $(repr(s_α.κ)) && kapB == $(repr(s_β.κ))")
        need_r && push!(conds, "rA == $(s_α.r) && rB == $(s_β.r)")
        keyword = first ? "if" : "elseif"
        tag = zm_mode === nothing ? "" : " ($zm_mode)"
        println(io, "$indent$keyword $(join(conds, " && "))")
        println(io, "$(indent2)# ⟨κA=$(_math_kappa(s_α.κ)), rA=$(s_α.r) | V | κB=$(_math_kappa(s_β.κ)), rB=$(s_β.r)⟩$tag")
        println(io, "$(indent2)error(\"TODO: fill matrix block for κA=$(_escape_str(s_α.κ)), κB=$(_escape_str(s_β.κ)), rA=$(s_α.r), rB=$(s_β.r)\")")
        first = false
    end
    println(io, "$(indent)end")
    println(io, "$(indent)error(\"unreachable: no matching branch for kapA=\$(repr(kapA)) kapB=\$(repr(kapB)) rA=\$rA rB=\$rB\")")
end

# ============ V-template helpers ============

function _need_branch_value(subs, field::Symbol)
    vals = Set{Any}()
    for s in subs
        push!(vals, field == :κ ? s.κ : s.r)
    end
    return length(vals) > 1
end

# Convert an arbitrary value to a representation embeddable in a Julia string literal (escape \" and \\).
function _escape_str(x)
    s = repr(x)
    s = replace(s, "\\" => "\\\\")
    s = replace(s, "\"" => "\\\"")
    return s
end
