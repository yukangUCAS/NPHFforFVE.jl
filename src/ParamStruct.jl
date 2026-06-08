# ============================================================
# ParamStruct — @params macro for JuMinuit-compatible parameter structs
# ============================================================
#
# Usage:
#
#     @params struct MyParams
#         C0 = 1.0       # leading-order contact
#         C1 = 0.5       # NLO contact
#         Λ  = 4.0       # cutoff (1/fm)
#     end
#
# This generates:
#
#   1. struct MyParams    — immutable type with fields C0, C1, Λ
#   2. MyParams(; C0=1.0, C1=0.5, Λ=4.0)   — keyword constructor with defaults
#   3. to_vector(p)       — [p.C0, p.C1, p.Λ]
#   4. from_vector(T, x)  — T(x[1], x[2], x[3])
#   5. param_names(T)     — ["C0", "C1", "Lambda"]
#   6. param_defaults(T)  — (C0=1.0, C1=0.5, Λ=4.0) (NamedTuple)
#
# Optionally override for your type:
#
#     function param_bounds(::Type{MyParams})
#         return [(0.0, Inf), (0.0, Inf), (0.0, Inf)]
#     end
#     function param_errors(::Type{MyParams})
#         return [0.1, 0.1, 0.1]
#     end
#
# ============================================================
# Integration with potential definitions:
#
# In potential_defs.jl, the V functions accept params as last argument:
#
#     function my_V_11(p, k, sp, s, kapA, kapB, rA, rB, aA, aB, params::MyParams)
#         return params.C0 + params.C1 * q_sq(p, k)
#     end
#
# ============================================================
# JuMinuit workflow:
#
#     p0 = MyParams()                          # defaults
#     x0 = to_vector(p0)                       # -> [1.0, 0.5, 4.0]
#     names = param_names(MyParams)             # -> ["C0", "C1", "Lambda"]
#     limits = param_limits(MyParams)           # -> [(0,nothing), nothing, ...]
#
#     m = Minuit(chi2_fcn, x0; names, errors=param_errors(MyParams), limits)
#     migrad!(m)
#     p_best = from_vector(MyParams, m.values)
# ============================================================
#
# ============================================================
#  宏实现 — 供内部使用，用户只需调用 @params
# ============================================================

# Generic function placeholders — @params adds specific methods
function to_vector end
function from_vector end
function param_names end
function param_defaults end

macro params(expr)
    if expr.head != :struct || length(expr.args) < 3
        error("@params must be used as: @params struct Name ... end")
    end
    name = expr.args[2]          # struct name
    if name isa Expr && name.head == :curly
        name = name.args[1]      # strip type params for now
    end
    body = expr.args[3]          # body block

    fields = Symbol[]
    defaults = Any[]

    for line in body.args
        if line isa LineNumberNode
            continue
        elseif line isa Expr && line.head == :(=)  # field = default
            fname = line.args[1]
            if fname isa Expr && fname.head == :(::)
                fname = fname.args[1]
            end
            push!(fields, fname)
            push!(defaults, line.args[2])
        elseif line isa Symbol  # field only, no default
            push!(fields, line)
            push!(defaults, nothing)
        end
    end

    # Build the struct expression
    field_exprs = [:( $f::Float64 ) for f in fields]

    # Build keyword constructor with defaults
    kw_args = []
    for (f, d) in zip(fields, defaults)
        if d !== nothing
            push!(kw_args, Expr(:kw, f, d))
        else
            push!(kw_args, Expr(:kw, f, 0.0))
        end
    end
    constructor = Expr(:function, Expr(:call, name, Expr(:parameters, kw_args...)),
                       Expr(:call, :new, [f for f in fields]...))

    # positional constructor (for from_vector)
    pos_ctor = Expr(:function, Expr(:call, name, [f for f in fields]...),
                    Expr(:call, :new, [f for f in fields]...))

    # to_vector
    to_vec_body = Expr(:vect, [:(p.$f) for f in fields]...)
    f_to_vec = Expr(:., __module__, QuoteNode(:to_vector))
    to_vec = Expr(:function, Expr(:call, f_to_vec, Expr(:(::), :p, name)),
                  to_vec_body)

    # from_vector
    from_vec_body = Expr(:call, name, [:(x[$i]) for (i, _) in enumerate(fields)]...)
    f_from_vec = Expr(:., __module__, QuoteNode(:from_vector))
    from_vec = Expr(:function, Expr(:call, f_from_vec,
                                    Expr(:(::), Expr(:curly, :Type, name)),
                                    Expr(:(::), :x, :(AbstractVector{<:Real}))),
                    from_vec_body)

    # param_names
    param_names_body = Expr(:vect, [String(f) for f in fields]...)
    f_param_names = Expr(:., __module__, QuoteNode(:param_names))
    param_names_fn = Expr(:function, Expr(:call, f_param_names,
                                          Expr(:(::), Expr(:curly, :Type, name))),
                          param_names_body)

    # param_defaults
    def_names = Expr(:tuple, [QuoteNode(f) for f in fields]...)
    def_vals = Expr(:tuple, [d !== nothing ? d : 0.0 for (f, d) in zip(fields, defaults)]...)
    f_param_def = Expr(:., __module__, QuoteNode(:param_defaults))
    param_def_fn = Expr(:function, Expr(:call, f_param_def,
                                        Expr(:(::), Expr(:curly, :Type, name))),
                        Expr(:call, Expr(:curly, :NamedTuple, def_names), def_vals))

    struct_ctor = Expr(:block,
        Expr(:struct, false, name, Expr(:block, field_exprs..., pos_ctor, constructor)),
        to_vec,
        from_vec,
        param_names_fn,
        param_def_fn,
    )

    return esc(struct_ctor)
end

# ============ JuMinuit 辅助函数（用户可按需覆盖） ============

"""
    param_bounds(::Type{T}) -> Vector{Tuple{Float64,Float64}}

返回参数边界 `[(lower, upper), ...]`。默认所有参数 (0.0, Inf)。
用户应为自己的参数类型覆盖此函数。
"""
function param_bounds(::Type{T}) where T
    names = param_names(T)
    return [(0.0, Inf) for _ in names]
end

"""
    param_errors(::Type{T}) -> Vector{Float64}

返回 JuMinuit 的初始步长。默认全部 0.1。
覆盖此函数以设置每个参数合适的初始步长。
"""
function param_errors(::Type{T}) where T
    return fill(0.1, param_count(T))
end

"""
    param_limits(::Type{T}) -> Vector

将 `param_bounds` 转换为 JuMinuit 的 limits 格式:
- 双边界: `(lo, up)`
- 无界: `nothing`
- 仅下界: `(lo, nothing)`
- 仅上界: `(nothing, up)`

可直接传入 `Minuit(fcn, x0; limits=param_limits(MyParams))`。
"""
function param_limits(::Type{T}) where T
    bounds = param_bounds(T)
    result = Vector{Any}(undef, length(bounds))
    for (i, (lo, up)) in enumerate(bounds)
        has_lo = isfinite(lo) && lo > -Inf
        has_up = isfinite(up) && up < Inf
        result[i] = if has_lo && has_up
            (lo, up)
        elseif has_lo
            (lo, nothing)
        elseif has_up
            (nothing, up)
        else
            nothing
        end
    end
    return result
end

# ============ 便利函数 ============

"""
    param_count(::Type{T}) -> Int

返回参数总数。
"""
param_count(::Type{T}) where T = length(param_names(T))

"""
    print_params(p)

打印参数名和当前值，方便查看。
"""
function print_params(p)
    T = typeof(p)
    v = to_vector(p)
    n = param_names(T)
    for i in 1:length(v)
        println("  $(n[i]) = $(v[i])")
    end
end

# ============ end of ParamStruct ============
