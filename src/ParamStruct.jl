# ============================================================
# ParamStruct — @params macro for potential and fit parameter structs
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
# ============================================================
# Integration with potential definitions:
#
# In potential_defs.jl, the V functions accept params as last argument:
#
#     function my_V_11(p, k, sp, s, kapA, kapB, rA, rB, params::MyParams)
#         return params.C0 + params.C1 * q_sq(p, k)
#     end
#
# ============================================================
# Macro implementation — users only need to call @params
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

    # Generate re-entrant safe code
    new_fields_tuple = Expr(:tuple, [QuoteNode(f) for f in fields]...)
    name_sym = QuoteNode(name)

    struct_and_helpers = Expr(:block,
        Expr(:struct, false, name, Expr(:block, field_exprs..., pos_ctor, constructor)),
        to_vec,
        from_vec,
        param_names_fn,
        param_def_fn,
    )

    safe_code = quote
        if isdefined(@__MODULE__, $name_sym)
            existing_fields = fieldnames($name)
            new_fields = $new_fields_tuple
            if existing_fields == new_fields
                # Fields are unchanged: safely skip the struct definition and refresh helper methods.
                $to_vec
                $from_vec
                $param_names_fn
                $param_def_fn
            else
                error("""
                    Cannot redefine struct $($name) with different fields.
                    Existing fields: $existing_fields
                    New fields:      $new_fields
                    → Please restart the Julia kernel and re-run.
                    """)
            end
        else
            $struct_and_helpers
        end
    end

    return esc(safe_code)
end

# ============ Convenience functions ============

"""
    param_count(::Type{T}) -> Int

Return the number of parameters.
"""
param_count(::Type{T}) where T = length(param_names(T))

"""
    print_params(p)

Print parameter names and their current values.
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
