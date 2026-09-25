"""A fitted interaction field computed from a config and fit parameters."""
struct DependentFitRule{Provider}
    required_fields::Tuple{Vararg{Symbol}}
    provider::Provider
end

"""Declare which fit fields a config-dependent provider uses."""
function depends_on(required_fields::Tuple{Vararg{Symbol}}, provider)
    length(unique(required_fields)) == length(required_fields) ||
        throw(ArgumentError("depends_on has duplicate fit fields"))
    return DependentFitRule(required_fields, provider)
end

"""A known interaction field computed from a config alone."""
struct KnownRule{Provider}
    provider::Provider
end

"""Declare a config-dependent field that is not fitted."""
known_from(provider) = KnownRule(provider)

"""Describe the source of every field in one config's interaction parameters."""
struct ParaMpiDependence{SharedFit,DependentFit,Known,Fixed}
    shared_fit::SharedFit
    dependent_fit::DependentFit
    known::Known
    fixed::Fixed
end

function ParaMpiDependence(; shared_fit=(), dependent_fit=NamedTuple(),
                           known=NamedTuple(), fixed=NamedTuple())
    shared_fit isa Tuple{Vararg{Symbol}} ||
        throw(ArgumentError("shared_fit must be a tuple of field names"))
    dependent_fit isa NamedTuple ||
        throw(ArgumentError("dependent_fit must be a NamedTuple"))
    known isa NamedTuple || throw(ArgumentError("known must be a NamedTuple"))
    fixed isa NamedTuple || throw(ArgumentError("fixed must be a NamedTuple"))
    all(rule -> rule isa DependentFitRule, values(dependent_fit)) ||
        throw(ArgumentError("dependent_fit values must use depends_on"))
    all(rule -> rule isa KnownRule, values(known)) ||
        throw(ArgumentError("known values must use known_from"))
    return ParaMpiDependence(shared_fit, dependent_fit, known, fixed)
end

"""Validated mapping from global fit parameters to interaction parameters."""
struct ParamMapping{InteractionParams,Dependence,FitParams}
    interaction_param_type::Type{InteractionParams}
    dependence::Dependence
    initial_fit_params::FitParams
end

function ParamMapping(interaction_param_type::Type{InteractionParams},
                                 dependence::ParaMpiDependence,
                                 initial_fit_params::FitParams) where {InteractionParams,FitParams}
    interaction_fields = fieldnames(InteractionParams)
    fit_fields = fieldnames(FitParams)
    sources = (dependence.shared_fit..., keys(dependence.dependent_fit)...,
               keys(dependence.known)..., keys(dependence.fixed)...)
    for field in interaction_fields
        count(==(field), sources) == 1 || throw(ArgumentError(
            "interaction field :$field must have exactly one source"))
    end
    for field in sources
        field in interaction_fields || throw(ArgumentError(
            "unknown interaction field :$field"))
    end
    for field in dependence.shared_fit
        field in fit_fields || throw(ArgumentError(
            "interaction field :$field requires missing fit field :$field"))
    end
    for (target, rule) in pairs(dependence.dependent_fit)
        for field in rule.required_fields
            field in fit_fields || throw(ArgumentError(
                "interaction field :$target requires missing fit field :$field"))
        end
    end
    return ParamMapping{InteractionParams,typeof(dependence),FitParams}(
        interaction_param_type, dependence, initial_fit_params)
end

function _resolved_interaction_value(value, config_index, field)
    value isa Real && !(value isa Bool) && isfinite(value) ||
        throw(ArgumentError(
            "config $config_index, interaction field :$field must resolve to a finite real number, got $value"))
    converted = Float64(value)
    isfinite(converted) || throw(ArgumentError(
        "config $config_index, interaction field :$field is outside the finite Float64 range"))
    return converted
end

"""Resolve one config's complete interaction parameters from global fit parameters."""
function resolve_params(pmap::ParamMapping{InteractionParams}, config::Config,
                        fit_params; config_index=nothing) where InteractionParams
    fit_params isa typeof(pmap.initial_fit_params) || throw(ArgumentError(
        "config $config_index requires fit parameters of type $(typeof(pmap.initial_fit_params))"))
    dependence = pmap.dependence
    values = map(fieldnames(InteractionParams)) do field
        value = try
            if field in dependence.shared_fit
                getproperty(fit_params, field)
            elseif haskey(dependence.dependent_fit, field)
                dependence.dependent_fit[field].provider(config, fit_params)
            elseif haskey(dependence.known, field)
                dependence.known[field].provider(config)
            else
                dependence.fixed[field]
            end
        catch err
            throw(ArgumentError(
                "config $config_index, interaction field :$field: $(sprint(showerror, err))"))
        end
        _resolved_interaction_value(value, config_index, field)
    end
    return InteractionParams(values...)
end

struct ConfigDependentRunParams{FitParams,DependenceSummary,InteractionParams}
    fit_params::FitParams
    dependence::DependenceSummary
    interaction_params::Vector{InteractionParams}
end

function _dependence_summary(dependence::ParaMpiDependence)
    dependent_fit = [(field=field,
                      required_fit_fields=rule.required_fields,
                      provider=_callable_label(rule.provider))
                     for (field, rule) in pairs(dependence.dependent_fit)]
    known = [(field=field, provider=_callable_label(rule.provider))
             for (field, rule) in pairs(dependence.known)]
    return (shared_fit=dependence.shared_fit,
            dependent_fit=dependent_fit,
            known=known,
            fixed=dependence.fixed)
end

function _write_parameter_info(io::IO, params::ConfigDependentRunParams)
    println(io, "# fit_parameter_type = ", typeof(params.fit_params))
    println(io, "# fit_parameters = ", repr(params.fit_params))
    println(io, "# dependence.shared_fit = ", repr(params.dependence.shared_fit))
    println(io, "# dependence.dependent_fit = ", repr(params.dependence.dependent_fit))
    println(io, "# dependence.known = ", repr(params.dependence.known))
    println(io, "# dependence.fixed = ", repr(params.dependence.fixed))
    for (config_index, interaction_params) in enumerate(params.interaction_params)
        println(io, "# interaction_parameters[$config_index] = ",
                repr(interaction_params))
    end
end

"""Compute each config with interaction parameters resolved from one fit point."""
function compute!(prepared::PreparedSpectrumProject, V_func,
                  pmap::ParamMapping{InteractionParams}, fit_params;
                  channel_filter=nothing, entry_filter=nothing,
                  verbose::Bool=false, channel_decomp::Bool=false,
                  validate_hermitian::Bool=false) where InteractionParams
    project = prepared.project
    n_configs = length(project.configs)
    spectra = Vector{Dict{String, Vector{Float64}}}(undef, n_configs)
    decomposition = [Dict{String, Vector{Dict{String, Float64}}}()
                     for _ in 1:n_configs]
    resolved_params = Vector{InteractionParams}(undef, n_configs)

    for (config_index, config) in enumerate(project.configs)
        interaction_params = resolve_params(
            pmap, config, fit_params; config_index=config_index)
        resolved_params[config_index] = interaction_params
        basis = prepared.groups[(config.L, config.a)]
        build_V_hel_blocks!(basis, V_func, interaction_params;
                            V_basis=project.V_basis,
                            channel_filter=channel_filter,
                            entry_filter=entry_filter,
                            validate_hermitian=validate_hermitian)
        verbose && println(basis)
        spectra[config_index] = _compute_prepared_config!(
            basis, config, prepared.backend,
            channel_decomp ? decomposition[config_index] : nothing)
    end

    metadata = ConfigDependentRunParams(
        fit_params, _dependence_summary(pmap.dependence), resolved_params)
    run_info = _run_info(project, metadata, V_func, prepared.backend,
                         :prepared_spectrum;
                         channel_filter=channel_filter,
                         entry_filter=entry_filter,
                         validate_hermitian=validate_hermitian,
                         channel_decomp=channel_decomp)
    return ProjectResult(run_info.project.configs, spectra, decomposition,
                         run_info)
end

"""Fit global parameters while resolving interaction parameters per config."""
function SpectrumFitProblem(prepared::PreparedSpectrumProject, V_func,
                            pmap::ParamMapping,
                            data::SpectrumDataset)
    _validate_fit_dataset(data, prepared.project)
    initial_fit_params = pmap.initial_fit_params
    names = _fit_parameter_metadata(initial_fit_params)
    evaluator = fit_params -> compute!(prepared, V_func, pmap, fit_params)
    loss = (result, fit_params) -> spectrum_chisq(result, data)
    return SpectrumFitProblem(evaluator, initial_fit_params, data, loss, names)
end

function SpectrumFitProblem(prepared::AffinePreparedProject, V_func,
                            pmap::ParamMapping,
                            data::SpectrumDataset)
    throw(ArgumentError(
        "ParamMapping requires prepare_spectrum; prepare_affine does not support config-dependent rules"))
end

"""Per-config affine caches for one parameter mapping (internal)."""
struct ConfigAffinePreparedProject{InteractionParams,Mapping}
    project::Project
    pmap::Mapping
    caches::Vector{AffinePreparedProject{InteractionParams}}
end

"""Prepare one existing affine cache per config using its known and fixed fields."""
function prepare_affine(project::Project, V_func,
                        pmap::ParamMapping{InteractionParams}; kwargs...) where InteractionParams
    isempty(project.configs) && throw(ArgumentError(
        "add at least one configuration with add_config! before computing"))
    dependence = pmap.dependence
    fixed_fields = (keys(dependence.known)..., keys(dependence.fixed)...)
    fixed_flags = (; (field => true for field in fixed_fields)...)
    caches = AffinePreparedProject{InteractionParams}[]
    prepared_project = deepcopy(project)
    for (config_index, config) in enumerate(prepared_project.configs)
        affine_probe_params = resolve_params(
            pmap, config, pmap.initial_fit_params; config_index=config_index)
        single_config_project = deepcopy(prepared_project)
        single_config_project.configs = [config]
        push!(caches, prepare_affine(single_config_project, V_func,
                                     affine_probe_params;
                                     fixed=fixed_flags, kwargs...))
    end
    return ConfigAffinePreparedProject{InteractionParams,typeof(pmap)}(
        prepared_project, pmap, caches)
end

"""Evaluate the cached affine interaction with fit parameters resolved per config."""
function compute!(prepared::ConfigAffinePreparedProject{InteractionParams},
                  fit_params; verbose::Bool=false,
                  channel_decomp::Bool=false) where InteractionParams
    project = prepared.project
    n_configs = length(project.configs)
    spectra = Vector{Dict{String, Vector{Float64}}}(undef, n_configs)
    decomposition = [Dict{String, Vector{Dict{String, Float64}}}()
                     for _ in 1:n_configs]
    resolved_params = Vector{InteractionParams}(undef, n_configs)

    for (config_index, config) in enumerate(project.configs)
        interaction_params = resolve_params(
            prepared.pmap, config, fit_params; config_index=config_index)
        resolved_params[config_index] = interaction_params
        one_result = compute!(prepared.caches[config_index], interaction_params;
                              verbose=verbose, channel_decomp=channel_decomp)
        spectra[config_index] = one_result.spectra[1]
        decomposition[config_index] = one_result.channel_decomp[1]
    end

    metadata = ConfigDependentRunParams(
        fit_params, _dependence_summary(prepared.pmap.dependence), resolved_params)
    first_cache = first(prepared.caches)
    run_info = ProjectRunInfo(
        deepcopy(project), deepcopy(metadata), first_cache.potential,
        first_cache.backend, :prepared_affine,
        first_cache.channel_filter, first_cache.entry_filter,
        first_cache.validate_hermitian, channel_decomp)
    return ProjectResult(run_info.project.configs, spectra, decomposition,
                         run_info)
end

"""Fit global parameters using the per-config affine caches."""
function SpectrumFitProblem(prepared::ConfigAffinePreparedProject,
                            initial_fit_params, data::SpectrumDataset)
    _validate_fit_dataset(data, prepared.project)
    typeof(initial_fit_params) === typeof(prepared.pmap.initial_fit_params) ||
        throw(ArgumentError("fit parameter type does not match ParamMapping"))
    names = _fit_parameter_metadata(initial_fit_params)
    evaluator = fit_params -> compute!(prepared, fit_params)
    loss = (result, fit_params) -> spectrum_chisq(result, data)
    return SpectrumFitProblem(evaluator, initial_fit_params, data, loss, names)
end
