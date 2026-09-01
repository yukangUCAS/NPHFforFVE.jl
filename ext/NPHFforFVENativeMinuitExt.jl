module NPHFforFVENativeMinuitExt

using NPHFforFVE
using NativeMinuit

function _ordered_setting(setting, names::Vector{String}, default,
                          label::AbstractString)
    setting === nothing && return fill(default, length(names))
    if setting isa AbstractVector
        length(setting) == length(names) || throw(DimensionMismatch(
            "$label has length $(length(setting)); expected $(length(names))"))
        return collect(setting)
    end
    setting isa NamedTuple || setting isa AbstractDict || throw(ArgumentError(
        "$label must be a NamedTuple, dictionary, vector, or nothing"))

    entries = Dict{String,Any}(String(k) => v for (k, v) in pairs(setting))
    unknown = setdiff(collect(keys(entries)), names)
    isempty(unknown) || throw(ArgumentError(
        "$label contains unknown parameter names: $(join(sort(unknown), ", "))"))
    return [get(entries, name, default) for name in names]
end

function _initial_steps(setting, names)
    values = Float64.(_ordered_setting(setting, names, 0.1,
                                       "initial_steps"))
    all(x -> isfinite(x) && x > 0, values) || throw(ArgumentError(
        "all initial_steps must be finite and positive"))
    return values
end

function _limit_entry(value, name::AbstractString)
    value === nothing && return nothing
    value isa Tuple && length(value) == 2 || throw(ArgumentError(
        "limit for $name must be nothing or a two-element tuple"))
    lower, upper = value
    (lower === nothing || lower isa Real) &&
        (upper === nothing || upper isa Real) || throw(ArgumentError(
            "limit endpoints for $name must be real numbers or nothing"))
    lower = lower === nothing || lower == -Inf ? nothing : Float64(lower)
    upper = upper === nothing || upper == Inf ? nothing : Float64(upper)
    lower === nothing || isfinite(lower) || throw(ArgumentError(
        "lower limit for $name must be finite, -Inf, or nothing"))
    upper === nothing || isfinite(upper) || throw(ArgumentError(
        "upper limit for $name must be finite, Inf, or nothing"))
    lower !== nothing && upper !== nothing && lower >= upper &&
        throw(ArgumentError("lower limit for $name must be less than upper limit"))
    return lower === nothing && upper === nothing ? nothing : (lower, upper)
end

function _limits(setting, names)
    raw = _ordered_setting(setting, names, nothing, "limits")
    return [_limit_entry(value, name) for (value, name) in zip(raw, names)]
end

function _fixed(setting, names)
    raw = _ordered_setting(setting, names, false, "fixed")
    all(value -> value isa Bool, raw) || throw(ArgumentError(
        "all fixed settings must be Bool"))
    return Vector{Bool}(raw)
end

function NPHFforFVE.minuit(problem::NPHFforFVE.SpectrumFitProblem;
                           initial_steps=nothing, limits=nothing,
                           fixed=nothing, threaded_gradient=false,
                           grad=nothing, kwargs...)
    threaded_gradient === false || throw(ArgumentError(
        "SpectrumFitProblem currently supports only threaded_gradient=false"))
    grad === nothing || throw(ArgumentError(
        "SpectrumFitProblem does not currently support a custom or AD gradient"))

    names = problem.parameter_names
    steps = _initial_steps(initial_steps, names)
    bounds = _limits(limits, names)
    fixed_values = _fixed(fixed, names)
    x0 = Float64.(NPHFforFVE.to_vector(problem.initial_params))
    for (i, (value, bound)) in enumerate(zip(x0, bounds))
        bound === nothing && continue
        lower, upper = bound
        lower !== nothing && value < lower && throw(ArgumentError(
            "initial value of $(names[i]) ($value) is below its lower limit ($lower)"))
        upper !== nothing && value > upper && throw(ArgumentError(
            "initial value of $(names[i]) ($value) is above its upper limit ($upper)"))
    end
    return NativeMinuit.Minuit(problem, x0;
                           names=names, errors=steps, limits=bounds,
                           fixed=fixed_values, threaded_gradient=false,
                           kwargs...)
end

function NPHFforFVE.best_params(problem::NPHFforFVE.SpectrumFitProblem,
                                fit::NativeMinuit.Minuit)
    fit_names = collect(String.(fit.parameters))
    fit_names == problem.parameter_names || throw(ArgumentError(
        "Minuit parameter names do not match the SpectrumFitProblem"))
    values = Float64.(collect(fit.values))
    length(values) == length(problem.parameter_names) || throw(DimensionMismatch(
        "Minuit has $(length(values)) values; expected " *
        "$(length(problem.parameter_names))"))
    return NPHFforFVE.from_vector(typeof(problem.initial_params), values)
end

end
