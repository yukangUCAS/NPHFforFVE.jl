# ============================================================
# IsospinSpace — Schur–Weyl duality: isospin × permutation-group decomposition
# ============================================================
#
# Input: particle count N; single-particle isospin j
# Output:
#   1. allowed S_N irreps [κ] and multiplicities
#   2. Yamanouchi orthogonal-representation matrices R^{[κ]}(p)
#   3. Charge states → |J,M,[κ],a⟩ linear-combination coefficients (CG coefficients)
#
# Convention: Yamanouchi orthogonal representation and Condon–Shortley phases.

using LinearAlgebra

# ============ S_N permutation-group elements ============

"""
    An S_N element represented by a 1-indexed permutation vector p, where p[i] is the destination of the particle at original position i.

Elements are ordered by BFS starting from generators s₁=(12), s₂=(23), ....
"""
const SN_ELEMENTS = Dict{Int, Vector{Vector{Int}}}()

# S_1: 1 elements
SN_ELEMENTS[1] = [
    [1],  # 1: identity e
]

# S_2: 2 elements
SN_ELEMENTS[2] = [
    [1, 2],  # 1: identity e
    [2, 1],  # 2: transposition (12)
]

# S_3: 6 elements (BFS from s₁=(12), s₂=(23))
SN_ELEMENTS[3] = [
    [1, 2, 3],  # 1: e
    [2, 1, 3],  # 2: (12) = s₁
    [1, 3, 2],  # 3: (23) = s₂
    [3, 1, 2],  # 4: (132) = s₁s₂
    [2, 3, 1],  # 5: (123) = s₂s₁
    [3, 2, 1],  # 6: (13) = s₁s₂s₁
]

# S_4: 24 elements (BFS from s₁=(12), s₂=(23), s₃=(34))
SN_ELEMENTS[4] = [
    [1, 2, 3, 4],  # 1: e
    [2, 1, 3, 4],  # 2: (12)
    [1, 3, 2, 4],  # 3: (23)
    [1, 2, 4, 3],  # 4: (34)
    [3, 1, 2, 4],  # 5: (132)
    [2, 1, 4, 3],  # 6: (12)(34)
    [2, 3, 1, 4],  # 7: (123)
    [1, 4, 2, 3],  # 8: (243)
    [1, 3, 4, 2],  # 9: (234)
    [3, 2, 1, 4],  # 10: (13)
    [4, 1, 2, 3],  # 11: (1432)
    [3, 1, 4, 2],  # 12: (1342)
    [2, 4, 1, 3],  # 13: (1243)
    [1, 4, 3, 2],  # 14: (24)
    [2, 3, 4, 1],  # 15: (1234)
    [4, 2, 1, 3],  # 16: (143)
    [4, 1, 3, 2],  # 17: (142)
    [3, 2, 4, 1],  # 18: (134)
    [3, 4, 1, 2],  # 19: (13)(24)
    [2, 4, 3, 1],  # 20: (124)
    [4, 3, 1, 2],  # 21: (1423)
    [4, 2, 3, 1],  # 22: (14)
    [3, 4, 2, 1],  # 23: (1324)
    [4, 3, 2, 1],  # 24: (14)(23)
]

# ============ S_N irrep matrices R^{[κ]}(p) ============

# Yamanouchi orthogonal-representation matrices, stored in SN_ELEMENTS order
const SN_MATRICES = Dict{Int, Dict{String, Vector{Matrix{Float64}}}}()

# ---- S_1 irreps (trivial) ----
SN_MATRICES[1] = Dict{String, Vector{Matrix{Float64}}}()
SN_MATRICES[1]["[1]"] = [
    fill(1.0, 1, 1),   # e
]

# ---- S_2 irreps (both 1D) ----
SN_MATRICES[2] = Dict{String, Vector{Matrix{Float64}}}()
SN_MATRICES[2]["[2]"] = [
    fill(1.0, 1, 1),   # e
    fill(1.0, 1, 1),   # (12)
]
SN_MATRICES[2]["[1,1]"] = [
    fill(1.0, 1, 1),   # e
    fill(-1.0, 1, 1),  # (12)
]

# ---- S_3 irreps ----
SN_MATRICES[3] = Dict{String, Vector{Matrix{Float64}}}()
# [3]: totally symmetric, 1D
SN_MATRICES[3]["[3]"] = [fill(1.0, 1, 1) for _ in 1:6]
# [2,1]: mixed symmetry, 2D Yamanouchi
SN_MATRICES[3]["[2,1]"] = [
    [1.0 0.0; 0.0 1.0],                                           # e
    [1.0 0.0; 0.0 -1.0],                                          # (12)
    [-0.5 0.8660254037844386; 0.8660254037844386 0.5],            # (23)
    [-0.5 -0.8660254037844386; 0.8660254037844386 -0.5],          # (132)
    [-0.5 0.8660254037844386; -0.8660254037844386 -0.5],          # (123)
    [-0.5 -0.8660254037844386; -0.8660254037844386 0.5],          # (13)
]

# [1,1,1]: totally antisymmetric, 1D — sign representation
SN_MATRICES[3]["[1,1,1]"] = [
    fill(1.0, 1, 1),    # e
    fill(-1.0, 1, 1),   # (12)
    fill(-1.0, 1, 1),   # (23)
    fill(1.0, 1, 1),    # (132)
    fill(1.0, 1, 1),    # (123)
    fill(-1.0, 1, 1),   # (13)
]

# ---- S_4 irreps ----
SN_MATRICES[4] = Dict{String, Vector{Matrix{Float64}}}()
# [4]: totally symmetric, 1D
SN_MATRICES[4]["[4]"] = [fill(1.0, 1, 1) for _ in 1:24]
# [3,1]: mixed, 3D Yamanouchi
SN_MATRICES[4]["[3,1]"] = [
    [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],                     # e
    [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 -1.0],                    # (12)
    [1.0 0.0 0.0; 0.0 -0.5 0.8660254037844386; 0.0 0.8660254037844386 0.5],  # (23)
    [-0.3333333333333333 0.9428090415820634 0.0; 0.9428090415820634 0.3333333333333333 0.0; 0.0 0.0 1.0],  # (34)
    [1.0 0.0 0.0; 0.0 -0.5 -0.8660254037844386; 0.0 0.8660254037844386 -0.5],  # (132)
    [-0.3333333333333333 0.9428090415820634 0.0; 0.9428090415820634 0.3333333333333333 0.0; 0.0 0.0 -1.0],  # (12)(34)
    [1.0 0.0 0.0; 0.0 -0.5 0.8660254037844386; 0.0 -0.8660254037844386 -0.5],  # (123)
    [-0.3333333333333333 -0.4714045207910317 0.8164965809277259; 0.9428090415820634 -0.16666666666666666 0.28867513459481287; 0.0 0.8660254037844386 0.5],  # (243)
    [-0.3333333333333333 0.9428090415820634 0.0; -0.4714045207910317 -0.16666666666666666 0.8660254037844386; 0.8164965809277259 0.28867513459481287 0.5],  # (234)
    [1.0 0.0 0.0; 0.0 -0.5 -0.8660254037844386; 0.0 -0.8660254037844386 0.5],  # (13)
    [-0.3333333333333333 -0.4714045207910317 -0.8164965809277259; 0.9428090415820634 -0.16666666666666666 -0.28867513459481287; 0.0 0.8660254037844386 -0.5],  # (1432)
    [-0.3333333333333333 0.9428090415820634 0.0; -0.4714045207910317 -0.16666666666666666 -0.8660254037844386; 0.8164965809277259 0.28867513459481287 -0.5],  # (1342)
    [-0.3333333333333333 -0.4714045207910317 0.8164965809277259; 0.9428090415820634 -0.16666666666666666 0.28867513459481287; 0.0 -0.8660254037844386 -0.5],  # (1243)
    [-0.3333333333333333 -0.4714045207910317 0.8164965809277259; -0.4714045207910317 0.8333333333333333 0.28867513459481287; 0.8164965809277259 0.28867513459481287 0.5],  # (24)
    [-0.3333333333333333 0.9428090415820634 0.0; -0.4714045207910317 -0.16666666666666666 0.8660254037844386; -0.8164965809277259 -0.28867513459481287 -0.5],  # (1234)
    [-0.3333333333333333 -0.4714045207910317 -0.8164965809277259; 0.9428090415820634 -0.16666666666666666 -0.28867513459481287; 0.0 -0.8660254037844386 0.5],  # (143)
    [-0.3333333333333333 -0.4714045207910317 -0.8164965809277259; -0.4714045207910317 0.8333333333333333 -0.28867513459481287; 0.8164965809277259 0.28867513459481287 -0.5],  # (142)
    [-0.3333333333333333 0.9428090415820634 0.0; -0.4714045207910317 -0.16666666666666666 -0.8660254037844386; -0.8164965809277259 -0.28867513459481287 0.5],  # (134)
    [-0.3333333333333333 -0.4714045207910317 0.8164965809277259; -0.4714045207910317 -0.6666666666666665 -0.5773502691896257; 0.8164965809277259 -0.5773502691896257 -0.0],  # (13)(24)
    [-0.3333333333333333 -0.4714045207910317 0.8164965809277259; -0.4714045207910317 0.8333333333333333 0.28867513459481287; -0.8164965809277259 -0.28867513459481287 -0.5],  # (124)
    [-0.3333333333333333 -0.4714045207910317 -0.8164965809277259; -0.4714045207910317 -0.6666666666666665 0.5773502691896257; 0.8164965809277259 -0.5773502691896257 0.0],  # (1423)
    [-0.3333333333333333 -0.4714045207910317 -0.8164965809277259; -0.4714045207910317 0.8333333333333333 -0.28867513459481287; -0.8164965809277259 -0.28867513459481287 0.5],  # (1432)?? actually it's (1432)
    [-0.3333333333333333 -0.4714045207910317 0.8164965809277259; -0.4714045207910317 -0.6666666666666665 -0.5773502691896257; -0.8164965809277259 0.5773502691896257 0.0],  # (1324)
    [-0.3333333333333333 -0.4714045207910317 -0.8164965809277259; -0.4714045207910317 -0.6666666666666665 0.5773502691896257; -0.8164965809277259 0.5773502691896257 -0.0],  # (14)(23)
]
# [2,2]: mixed, 2D Yamanouchi
SN_MATRICES[4]["[2,2]"] = [
    [1.0 0.0; 0.0 1.0],                                            # e
    [1.0 0.0; 0.0 -1.0],                                           # (12)
    [-0.5 0.8660254037844386; 0.8660254037844386 0.5],             # (23)
    [1.0 0.0; 0.0 -1.0],                                           # (34)
    [-0.5 -0.8660254037844386; 0.8660254037844386 -0.5],           # (132)
    [1.0 0.0; 0.0 1.0],                                            # (12)(34)
    [-0.5 0.8660254037844386; -0.8660254037844386 -0.5],           # (123)
    [-0.5 0.8660254037844386; -0.8660254037844386 -0.5],           # (243)
    [-0.5 -0.8660254037844386; 0.8660254037844386 -0.5],           # (234)
    [-0.5 -0.8660254037844386; -0.8660254037844386 0.5],           # (13)
    [-0.5 -0.8660254037844386; -0.8660254037844386 0.5],           # (1432)
    [-0.5 0.8660254037844386; 0.8660254037844386 0.5],             # (1342)
    [-0.5 0.8660254037844386; 0.8660254037844386 0.5],             # (1243)
    [-0.5 -0.8660254037844386; -0.8660254037844386 0.5],           # (24)
    [-0.5 -0.8660254037844386; -0.8660254037844386 0.5],           # (1234)
    [-0.5 -0.8660254037844386; 0.8660254037844386 -0.5],           # (143)
    [-0.5 0.8660254037844386; -0.8660254037844386 -0.5],           # (142)
    [-0.5 0.8660254037844386; -0.8660254037844386 -0.5],           # (134)
    [1.0 0.0; 0.0 1.0],                                            # (13)(24)
    [-0.5 -0.8660254037844386; 0.8660254037844386 -0.5],           # (124)
    [1.0 0.0; 0.0 -1.0],                                           # (1423)
    [-0.5 0.8660254037844386; 0.8660254037844386 0.5],             # (1432) actually
    [1.0 0.0; 0.0 -1.0],                                           # (1324)
    [1.0 0.0; 0.0 1.0],                                            # (14)(23)
]
# [2,1,1]: mixed, 3D Yamanouchi
SN_MATRICES[4]["[2,1,1]"] = [
    [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],                                           # e
    [1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 -1.0],                                          # (12)
    [-0.5 0.8660254037844386 0.0; 0.8660254037844386 0.5 0.0; 0.0 0.0 -1.0],            # (23)
    [-1.0 0.0 0.0; 0.0 -0.3333333333333333 0.9428090415820635; 0.0 0.9428090415820635 0.3333333333333333],  # (34)
    [-0.5 -0.8660254037844386 0.0; 0.8660254037844386 -0.5 0.0; 0.0 0.0 1.0],            # (132)
    [-1.0 0.0 0.0; 0.0 0.3333333333333333 -0.9428090415820635; 0.0 -0.9428090415820635 -0.3333333333333333],  # (12)(34)
    [-0.5 0.8660254037844386 0.0; -0.8660254037844386 -0.5 0.0; 0.0 0.0 1.0],            # (123)
    [0.5 -0.8660254037844386 0.0; -0.28867513459481287 -0.16666666666666666 -0.9428090415820635; 0.816496580927726 0.47140452079103173 -0.3333333333333333],  # (243)
    [0.5 -0.28867513459481287 0.816496580927726; -0.8660254037844386 -0.16666666666666666 0.47140452079103173; 0.0 -0.9428090415820635 -0.3333333333333333],  # (234)
    [-0.5 -0.8660254037844386 0.0; -0.8660254037844386 0.5 0.0; 0.0 0.0 -1.0],            # (13)
    [0.5 0.8660254037844386 0.0; -0.28867513459481287 0.16666666666666666 0.9428090415820635; 0.816496580927726 -0.47140452079103173 0.3333333333333333],  # (1432)
    [0.5 0.28867513459481287 -0.816496580927726; -0.8660254037844386 0.16666666666666666 -0.47140452079103173; 0.0 0.9428090415820635 0.3333333333333333],  # (1342)
    [0.5 -0.8660254037844386 0.0; 0.28867513459481287 0.16666666666666666 0.9428090415820635; -0.816496580927726 -0.47140452079103173 0.3333333333333333],  # (1243)
    [-0.5 0.28867513459481287 -0.816496580927726; 0.28867513459481287 -0.8333333333333333 -0.47140452079103173; -0.816496580927726 -0.47140452079103173 0.3333333333333333],  # (24)
    [0.5 -0.28867513459481287 0.816496580927726; 0.8660254037844386 0.16666666666666666 -0.47140452079103173; 0.0 0.9428090415820635 0.3333333333333333],  # (1234)
    [0.5 0.8660254037844386 0.0; 0.28867513459481287 -0.16666666666666666 -0.9428090415820635; -0.816496580927726 0.47140452079103173 -0.3333333333333333],  # (143)
    [-0.5 -0.28867513459481287 0.816496580927726; 0.28867513459481287 0.8333333333333333 0.47140452079103173; -0.816496580927726 0.47140452079103173 -0.3333333333333333],  # (142)
    [0.5 0.28867513459481287 -0.816496580927726; 0.8660254037844386 -0.16666666666666666 0.47140452079103173; 0.0 -0.9428090415820635 -0.3333333333333333],  # (134)
    [0.0 0.5773502691896257 0.816496580927726; 0.5773502691896257 -0.6666666666666665 0.47140452079103173; 0.816496580927726 0.47140452079103173 -0.3333333333333333],  # (13)(24)
    [-0.5 0.28867513459481287 -0.816496580927726; -0.28867513459481287 0.8333333333333333 0.47140452079103173; 0.816496580927726 0.47140452079103173 -0.3333333333333333],  # (124)
    [0.0 -0.5773502691896257 -0.816496580927726; 0.5773502691896257 0.6666666666666665 -0.47140452079103173; 0.816496580927726 -0.47140452079103173 0.3333333333333333],  # (1423)
    [-0.5 -0.28867513459481287 0.816496580927726; -0.28867513459481287 -0.8333333333333333 -0.47140452079103173; 0.816496580927726 -0.47140452079103173 0.3333333333333333],  # (1432)
    [0.0 0.5773502691896257 0.816496580927726; -0.5773502691896257 0.6666666666666665 -0.47140452079103173; -0.816496580927726 -0.47140452079103173 0.3333333333333333],  # (1324)
    [0.0 -0.5773502691896257 -0.816496580927726; -0.5773502691896257 -0.6666666666666665 0.47140452079103173; -0.816496580927726 0.47140452079103173 -0.3333333333333333],  # (14)(23)
]

# ============ S_N irrep query functions ============

"""
    get_SN_irrep_names(N::Int) -> Vector{String}

Return all currently supported S_N irrep names for N particles.
"""
function get_SN_irrep_names(N::Int)
    if N == 1
        return ["[1]"]
    elseif N == 2
        return ["[2]", "[1,1]"]
    elseif N == 3
        return ["[3]", "[2,1]", "[1,1,1]"]
    elseif N == 4
        return ["[4]", "[3,1]", "[2,2]", "[2,1,1]"]
    else
        throw(ArgumentError("S_$N irreps not yet implemented"))
    end
end

"""
    get_SN_irrep_dim(N::Int, irrep::String) -> Int

Return the dimension of irrep [κ].
"""
function get_SN_irrep_dim(N::Int, irrep::String)
    mats = get_SN_irrep_matrices(N, irrep)
    return size(mats[1], 1)
end

"""
    get_SN_irrep_matrices(N::Int, irrep::String) -> Vector{Matrix{Float64}}

Return Yamanouchi orthogonal-representation matrices R^{[κ]}(p_i) of S_N irrep [κ] at all group elements.
Index i corresponds to SN_ELEMENTS[N][i].
"""
function get_SN_irrep_matrices(N::Int, irrep::String)
    if haskey(SN_MATRICES, N) && haskey(SN_MATRICES[N], irrep)
        return SN_MATRICES[N][irrep]
    else
        throw(ArgumentError("S_$N irrep matrices not available for $irrep"))
    end
end

"""
    get_SN_irrep_matrix(N::Int, irrep::String, i::Int) -> Matrix{Float64}

Return R^{[κ]}(p_i), the representation matrix at the i-th element.
"""
function get_SN_irrep_matrix(N::Int, irrep::String, i::Int)
    return get_SN_irrep_matrices(N, irrep)[i]
end

"""
    get_SN_element_index(N::Int, p::Vector{Int}) -> Int

Return the index of permutation vector p in SN_ELEMENTS[N].
"""
function get_SN_element_index(N::Int, p::Vector{Int})
    for (i, el) in enumerate(SN_ELEMENTS[N])
        if el == p
            return i
        end
    end
    throw(ArgumentError("Permutation $p not found in S_$N elements"))
end

# ============ Isospin decomposition tables ============

"""
    IsospinDecomposition

Store isospin decomposition data for a given (j, N).

Fields:
- j: single-particle isospin
- N: particle count
- entries: [(J, [κ], multiplicity), ...] in descending J order
  where multiplicity is that of S_N irrep [κ] at total isospin J
  (Yamanouchi basis index a=1,…,dim([κ]) is a representation-matrix row index, distinct from multiplicity).
"""
struct IsospinDecomposition
    j::Rational{Int}
    N::Int
    entries::Vector{Tuple{Rational{Int}, String, Int}}  # (J, [κ], multiplicity)
end

function Base.show(io::IO, d::IsospinDecomposition)
    println(io, "IsospinDecomposition (N=$(d.N), j=$(d.j)):")
    println(io, "  J    [κ]       multiplicity")
    println(io, "  ---- --------- ------------")
    for (J, irrep, mult) in d.entries
        println(io, "  $(lpad(string(J), 4)) $(rpad(irrep, 9)) $mult")
    end
end

"""
    isospin_decomposition(N::Int, j::Union{Rational{Int}, Int}) -> IsospinDecomposition

Return the Schur–Weyl decomposition of N particles with isospin j.

Return all triples (total isospin J, S_N irrep [κ], multiplicity).
"""
function isospin_decomposition(N::Int, j::Union{Rational{Int}, Int})
    j_r = Rational{Int}(j)
    if N == 2 && j_r == 1//2
        return IsospinDecomposition(j_r, N, [
            (1//1, "[2]",    1),   # J=1, [2] symmetric, triplet
            (0//1, "[1,1]",  1),   # J=0, [1,1] antisymmetric, singlet
        ])
    elseif N == 3 && j_r == 1//2
        return IsospinDecomposition(j_r, N, [
            (3//2, "[3]",    1),   # J=3/2, [3] fully symmetric
            (1//2, "[2,1]",  1),   # J=1/2, [2,1] mixed symmetry, dim=2
        ])
    elseif N == 4 && j_r == 1//2
        return IsospinDecomposition(j_r, N, [
            (2//1, "[4]",    1),   # J=2, [4] fully symmetric
            (1//1, "[3,1]",  1),   # J=1, [3,1] mixed symmetry, dim=3
            (0//1, "[2,2]",  1),   # J=0, [2,2] mixed symmetry, dim=2
        ])
    elseif N == 2 && j_r == 1//1
        return IsospinDecomposition(j_r, N, [
            (2//1, "[2]",    1),   # J=2, [2] symmetric
            (1//1, "[1,1]",  1),   # J=1, [1,1] antisymmetric
            (0//1, "[2]",    1),   # J=0, [2] symmetric
        ])
    elseif N == 2 && j_r == 3//2
        return IsospinDecomposition(j_r, N, [
            (3//1, "[2]",    1),   # (-1)^(2j-J) = +1
            (2//1, "[1,1]",  1),   # (-1)^(2j-J) = -1
            (1//1, "[2]",    1),
            (0//1, "[1,1]",  1),
        ])
    elseif N == 3 && j_r == 1//1
        return IsospinDecomposition(j_r, N, [
            (3//1, "[3]",       1),   # J=3, [3] fully symmetric
            (2//1, "[2,1]",     1),   # J=2, [2,1] mixed symmetry, dim=2
            (1//1, "[3]",       1),   # J=1, [3] + [2,1]
            (1//1, "[2,1]",     1),
            (0//1, "[1,1,1]",   1),   # J=0, [1,1,1] fully antisymmetric
        ])
    elseif N == 4 && j_r == 1//1
        return IsospinDecomposition(j_r, N, [
            (4//1, "[4]",       1),   # J=4, [4] fully symmetric
            (3//1, "[3,1]",     1),   # J=3, [3,1] mixed symmetry, dim=3
            (2//1, "[4]",       1),   # J=2, [4] + [3,1] + [2,2]
            (2//1, "[3,1]",     1),
            (2//1, "[2,2]",     1),
            (1//1, "[3,1]",     1),   # J=1, [3,1] + [2,1,1]
            (1//1, "[2,1,1]",   1),
            (0//1, "[4]",       1),   # J=0, [4] + [2,2]
            (0//1, "[2,2]",     1),
        ])
    elseif N == 1
        return IsospinDecomposition(j_r, N, [(j_r, "[1]", 1)])
    else
        throw(ArgumentError("Isospin decomposition not yet implemented for N=$N, j=$j_r"))
    end
end

# ============ CG coefficients: charge states → |J,M,[κ],a⟩ ============

"""
    ChargeToIsospinCG

Store CG coefficients for given (j, N, J, [κ]).

|J,M,[κ],a⟩ = Σ_{m₁,...,m_N} coeff[(m₁,...,m_N)] |m₁,...,m_N⟩

Here a = 1,...,dim([κ]) is the Yamanouchi-basis row index of [κ].
"""
struct ChargeToIsospinCG{N}
    j::Rational{Int}
    J::Rational{Int}
    irrep::String    # [κ]
    multiplicity::Int
    dim_irrep::Int   # dim([κ])
    # coeffs[M][a] = Dict{m_tuple => coefficient}
    coeffs::Dict{Rational{Int}, Dict{Int, Dict{NTuple{N, Rational{Int}}, ComplexF64}}}
end

function Base.show(io::IO, cg::ChargeToIsospinCG{N}) where {N}
    n_states = sum(length(d) for M_dict in values(cg.coeffs) for a_dict in values(M_dict))
    println(io, "ChargeToIsospinCG (N=$N, j=$(cg.j), J=$(cg.J), [κ]=$(cg.irrep), mult=$(cg.multiplicity)): $n_states states")
end

# ---- Helper functions: construct CG coefficients ----

function _build_cg_coeffs(::Val{N}, j::Rational{Int}, J::Rational{Int},
                           irrep::String, multiplicity::Int, dim_irrep::Int,
                           terms_list) where {N}
    # terms_list: [(M, a, [(m_tuple, coeff_real), ...]), ...]
    coeffs = Dict{Rational{Int}, Dict{Int, Dict{NTuple{N, Rational{Int}}, ComplexF64}}}()
    for (M, a, term_vec) in terms_list
        if !haskey(coeffs, M)
            coeffs[M] = Dict{Int, Dict{NTuple{N, Rational{Int}}, ComplexF64}}()
        end
        if !haskey(coeffs[M], a)
            coeffs[M][a] = Dict{NTuple{N, Rational{Int}}, ComplexF64}()
        end
        for (mt, c) in term_vec
            coeffs[M][a][mt] = ComplexF64(c, 0.0)
        end
    end
    cg = ChargeToIsospinCG{N}(j, J, irrep, multiplicity, dim_irrep, coeffs)
    _fix_SU2_lowering!(cg, j)
    _fix_SN_signs!(cg)
    return cg
end

function _lowering_coeff(j_r::Rational{Int}, m::Rational{Int})
    # J_- |j,m⟩ = √(j(j+1)-m(m-1)) |j,m-1⟩
    return sqrt(Float64(j_r*(j_r+1) - m*(m-1)))
end

function _apply_J_lowering(m_config::NTuple{N,Rational{Int}}, j_r::Rational{Int}) where {N}
    result = Dict{NTuple{N,Rational{Int}}, Float64}()
    for i in 1:N
        m_i = m_config[i]
        c = _lowering_coeff(j_r, m_i)
        if abs(c) > 1e-14
            new_cfg = collect(m_config)
            new_cfg[i] = m_i - 1
            key = NTuple{N,Rational{Int}}(new_cfg)
            result[key] = get(result, key, 0.0) + c
        end
    end
    return result
end

function _fix_SU2_lowering!(cg::ChargeToIsospinCG{N}, j::Rational{Int}) where {N}
    M_values = sort(collect(keys(cg.coeffs)))
    length(M_values) <= 1 && return cg  # Only one M value, nothing to lower

    J_val = cg.J
    M_max = M_values[end]

    # For each M from J-1 down to -J, regenerate from M+1 via lowering
    for k in 1:length(M_values)-1
        M_upper = M_values[end - k + 1]
        M_lower = M_values[end - k]

        expected_factor = Float64(sqrt(J_val*(J_val+1) - M_upper*(M_upper-1)))
        a_keys = sort(collect(keys(cg.coeffs[M_upper])))

        new_lower_dict = Dict{Int, Dict{NTuple{N,Rational{Int}}, ComplexF64}}()
        for a in a_keys
            lowered = Dict{NTuple{N,Rational{Int}}, Float64}()
            for (mc, coeff) in cg.coeffs[M_upper][a]
                for (mc_new, lc) in _apply_J_lowering(mc, j)
                    lowered[mc_new] = get(lowered, mc_new, 0.0) + real(coeff) * lc
                end
            end

            # Normalize: divide by expected_factor
            new_coeff_dict = Dict{NTuple{N,Rational{Int}}, ComplexF64}()
            for (mc, val) in lowered
                new_coeff_dict[mc] = ComplexF64(val / expected_factor, 0.0)
            end
            new_lower_dict[a] = new_coeff_dict
        end
        cg.coeffs[M_lower] = new_lower_dict
    end

    return cg
end

function _apply_permutation_to_config(m_config::NTuple{N,Rational{Int}}, perm::Vector{Int}) where {N}
    perm_inv = similar(perm)
    for i in eachindex(perm)
        perm_inv[perm[i]] = i
    end
    return ntuple(i -> m_config[perm_inv[i]], N)
end

function _fix_SN_signs!(cg::ChargeToIsospinCG{N}) where {N}
    dim_irrep = cg.dim_irrep
    dim_irrep <= 1 && return cg  # 1D irreps always correct

    D_ref = get_SN_irrep_matrices(N, cg.irrep)
    SN_list = SN_ELEMENTS[N]

    M_values = sort(collect(keys(cg.coeffs)))
    M_max = M_values[end]  # J

    for M in M_values
        a_keys = sort(collect(keys(cg.coeffs[M])))
        length(a_keys) != dim_irrep && continue  # sanity check

        # Compute S_N matrices from CG coefficients
        mats = Vector{Matrix{Float64}}(undef, length(SN_list))
        for (p_idx, perm_vec) in enumerate(SN_list)
            mat = zeros(Float64, dim_irrep, dim_irrep)
            for (ai, a) in enumerate(a_keys)
                coeff_a = cg.coeffs[M][a]
                for (ab, b) in enumerate(a_keys)
                    dotprod = 0.0
                    for (mc, cb) in cg.coeffs[M][b]
                        mc_perm = _apply_permutation_to_config(mc, perm_vec)
                        if haskey(coeff_a, mc_perm)
                            dotprod += cb * coeff_a[mc_perm]
                        end
                    end
                    mat[ai, ab] = dotprod
                end
            end
            mats[p_idx] = mat
        end

        # Check if already correct
        if all(mats[i] ≈ D_ref[i] for i in 1:length(D_ref))
            continue
        end

        # Find diagonal ±1 fix
        found = false
        for combo in 0:(2^dim_irrep - 1)
            signs = [((combo >> k) & 1) == 1 ? -1.0 : 1.0 for k in 0:(dim_irrep-1)]
            T = diagm(signs)

            all_match = true
            for i in 1:length(D_ref)
                if !(T * mats[i] * T ≈ D_ref[i])
                    all_match = false
                    break
                end
            end

            if all_match
                # Apply transformation to CG coefficients
                new_M_dict = Dict{Int, Dict{NTuple{N, Rational{Int}}, ComplexF64}}()
                for (a_new_idx, a_new) in enumerate(a_keys)
                    new_coeff_dict = Dict{NTuple{N, Rational{Int}}, ComplexF64}()
                    for (a_old_idx, a_old) in enumerate(a_keys)
                        factor = signs[a_new_idx] * (a_new_idx == a_old_idx ? 1.0 : 0.0)  # T is diagonal
                        # Actually T_{a_new, a_old} = signs[a_new] if a_new == a_old, else 0
                        # But we used T = diagm(signs), and T * mats * T = D_ref
                        # So the basis transforms as |a'_new⟩ = sum_a T_{a_new, a} |a⟩
                        # where we stored signs as T = diagm(signs)
                        # So T_{a_new, a} = signs[a_new] * delta_{a_new, a}
                        if a_new == a_old && abs(signs[a_new_idx] - 1.0) < 1e-10
                            new_coeff_dict = copy(cg.coeffs[M][a_new])
                        elseif a_new == a_old  # signs[a_new_idx] == -1.0
                            for (mc, coeff) in cg.coeffs[M][a_new]
                                new_coeff_dict[mc] = -coeff
                            end
                        end
                    end
                    new_M_dict[a_new] = new_coeff_dict
                end
                cg.coeffs[M] = new_M_dict
                found = true
                break
            end
        end

        if !found
            # Try permutation + signs
            perms = dim_irrep == 2 ? [[1,2],[2,1]] :
                    [[1,2,3],[2,1,3],[1,3,2],[3,2,1],[2,3,1],[3,1,2]]
            found2 = false
            for pp in perms
                P = zeros(dim_irrep, dim_irrep)
                for i in 1:dim_irrep
                    P[i, pp[i]] = 1.0
                end
                for combo in 0:(2^dim_irrep - 1)
                    signs = [((combo >> k) & 1) == 1 ? -1.0 : 1.0 for k in 0:(dim_irrep-1)]
                    T = diagm(signs) * P
                    Tinv = P' * diagm(signs)

                    all_match = true
                    for i in 1:length(D_ref)
                        if !(Tinv * mats[i] * T ≈ D_ref[i])
                            all_match = false
                            break
                        end
                    end

                    if all_match
                        # Apply transformation
                        new_M_dict = Dict{Int, Dict{NTuple{N, Rational{Int}}, ComplexF64}}()
                        for (a_new_idx, a_new) in enumerate(a_keys)
                            new_coeff_dict = Dict{NTuple{N, Rational{Int}}, ComplexF64}()
                            for (a_old_idx, a_old) in enumerate(a_keys)
                                factor = T[a_new_idx, a_old_idx]
                                if abs(factor) > 1e-10
                                    for (mc, coeff) in cg.coeffs[M][a_old]
                                        new_coeff_dict[mc] = get(new_coeff_dict, mc, zero(ComplexF64)) + factor * coeff
                                    end
                                end
                            end
                            new_M_dict[a_new] = new_coeff_dict
                        end
                        cg.coeffs[M] = new_M_dict
                        found2 = true
                        break
                    end
                end
                if found2
                    break
                end
            end
            @assert found2 "Cannot fix S_N signs for N=$N j=$(cg.j) J=$(cg.J) $(cg.irrep) M=$M"
        end
    end
    return cg
end

"""
    charge_to_isospin_cg(N::Int, j::Rational{Int}, J::Rational{Int}, irrep::String; multiplicity::Int=1)
        -> ChargeToIsospinCG

Return the CG coefficient from charge basis |m₁,...,m_N⟩ to |J,M,[κ],a⟩.
"""
function charge_to_isospin_cg(N::Int, j::Union{Rational{Int},Int}, J::Union{Rational{Int},Int},
                                irrep::String; multiplicity::Int=1)
    j_r = Rational{Int}(j)
    J_r = Rational{Int}(J)
    dim_irrep = get_SN_irrep_dim(N, irrep)

    if N == 2 && j_r == 3//2
        multiplicity == 1 || throw(ArgumentError(
            "N=2, j=3/2 has no multiplicity copy $multiplicity"))
        expected_irrep = J_r in (3//1, 1//1) ? "[2]" :
                         J_r in (2//1, 0//1) ? "[1,1]" : nothing
        expected_irrep === nothing && throw(ArgumentError(
            "No such total J=$J_r for N=2, j=3/2"))
        irrep == expected_irrep || throw(ArgumentError(
            "For N=2, j=3/2, J=$J_r requires κ=$expected_irrep, got κ=$irrep"))
        terms_list = []
        m_values = collect((-j_r):1:j_r)
        for M in (-J_r):1:J_r
            terms = []
            for m1 in m_values
                m2 = M - m1
                m2 in m_values || continue
                coefficient = _su2_clebsch_gordan(
                    j_r, m1, j_r, m2, J_r, M)
                abs(coefficient) > 1e-14 &&
                    push!(terms, ((m1, m2), coefficient))
            end
            push!(terms_list, (M, 1, terms))
        end
        return _build_cg_coeffs(
            Val(2), j_r, J_r, irrep, multiplicity, dim_irrep, terms_list)

    elseif N == 2 && j_r == 1//2
        up, down = 1//2, -1//2
        r2inv = 1.0 / sqrt(2.0)

        if J_r == 1//1 && irrep == "[2]"
            return _build_cg_coeffs(Val(2), j_r, J_r, irrep, multiplicity, dim_irrep, [
                (-1//1, 1, [( (down, down), 1.0 )]),
                ( 0//1, 1, [( (down, up),   r2inv ),
                             ( (up, down),   r2inv )]),
                ( 1//1, 1, [( (up, up),     1.0 )]),
            ])
        elseif J_r == 0//1 && irrep == "[1,1]"
            return _build_cg_coeffs(Val(2), j_r, J_r, irrep, multiplicity, dim_irrep, [
                (0//1, 1, [( (up, down),    r2inv ),
                            ( (down, up),  -r2inv )]),
            ])
        else
            throw(ArgumentError("No such (J=$J_r, [κ]=$irrep) for N=2, j=1/2"))
        end

    elseif N == 3 && j_r == 1//2
        up, down = 1//2, -1//2
        r3inv = 1.0 / sqrt(3.0)
        r6inv = 1.0 / sqrt(6.0)
        r2inv = 1.0 / sqrt(2.0)
        s23 = sqrt(2.0/3.0)

        if J_r == 3//2 && irrep == "[3]"
            # [3] fully symmetric, a=1 only
            return _build_cg_coeffs(Val(3), j_r, J_r, irrep, multiplicity, dim_irrep, [
                (-3//2, 1, [( (down, down, down),  1.0 )]),
                (-1//2, 1, [( (down, down, up),   -r3inv ),
                             ( (up, down, down),   -r3inv ),
                             ( (down, up, down),   -r3inv )]),
                ( 1//2, 1, [( (down, up, up),     -r3inv ),
                             ( (up, up, down),    -r3inv ),
                             ( (up, down, up),    -r3inv )]),
                ( 3//2, 1, [( (up, up, up),        1.0 )]),
            ])
        elseif J_r == 1//2 && irrep == "[2,1]"
            # [2,1] Yamanouchi, a=1 (T1), a=2 (T2)
            return _build_cg_coeffs(Val(3), j_r, J_r, irrep, multiplicity, dim_irrep, [
                # a=1 (T1, s₁ eigenvalue +1)
                (-1//2, 1, [( (down, down, up),    s23 ),
                             ( (down, up, down),  -r6inv ),
                             ( (up, down, down),  -r6inv )]),
                ( 1//2, 1, [( (up, up, down),     -s23 ),
                             ( (up, down, up),     r6inv ),
                             ( (down, up, up),     r6inv )]),
                # a=2 (T2, s₁ eigenvalue -1)
                (-1//2, 2, [( (up, down, down),    r2inv ),
                             ( (down, up, down),  -r2inv )]),
                ( 1//2, 2, [( (up, down, up),     -r2inv ),
                             ( (down, up, up),     r2inv )]),
            ])
        else
            throw(ArgumentError("No such (J=$J_r, [κ]=$irrep) for N=3, j=1/2"))
        end

    elseif N == 4 && j_r == 1//2
        up, down = 1//2, -1//2
        r2inv = 1.0 / sqrt(2.0)
        r3inv = 1.0 / sqrt(3.0)
        r6inv = 1.0 / sqrt(6.0)
        r12inv = 1.0 / sqrt(12.0)
        s34 = sqrt(3.0/4.0)
        s23 = sqrt(2.0/3.0)
        s112 = sqrt(1.0/12.0)

        if J_r == 2//1 && irrep == "[4]"
            return _build_cg_coeffs(Val(4), j_r, J_r, irrep, multiplicity, dim_irrep, [
                (-2//1, 1, [( (down, down, down, down),  1.0 )]),
                (-1//1, 1, [( (down, down, down, up),  -0.5 ),
                             ( (up, down, down, down),  -0.5 ),
                             ( (down, up, down, down),  -0.5 ),
                             ( (down, down, up, down),  -0.5 )]),
                ( 0//1, 1, [( (down, up, down, up),    -r6inv ),
                             ( (up, up, down, down),   -r6inv ),
                             ( (up, down, up, down),   -r6inv ),
                             ( (down, up, up, down),   -r6inv ),
                             ( (up, down, down, up),   -r6inv ),
                             ( (down, down, up, up),   -r6inv )]),
                ( 1//1, 1, [( (up, up, up, down),      -0.5 ),
                             ( (up, up, down, up),     -0.5 ),
                             ( (up, down, up, up),     -0.5 ),
                             ( (down, up, up, up),     -0.5 )]),
                ( 2//1, 1, [( (up, up, up, up),         1.0 )]),
            ])
        elseif J_r == 1//1 && irrep == "[3,1]"
            return _build_cg_coeffs(Val(4), j_r, J_r, irrep, multiplicity, dim_irrep, [
                # a=1 (T1, s₁=+1, s₂=+1)
                (-1//1, 1, [( (down, down, down, up),  -s34 ),
                              ( (down, up, down, down),  s112 ),
                              ( (down, down, up, down),  s112 ),
                              ( (up, down, down, down),  s112 )]),
                ( 0//1, 1, [( (down, down, up, up),    -r6inv ),
                              ( (up, down, up, down),    r6inv ),
                              ( (down, up, up, down),    r6inv ),
                              ( (up, down, down, up),   -r6inv ),
                              ( (down, up, down, up),   -r6inv ),
                              ( (up, up, down, down),    r6inv )]),
                ( 1//1, 1, [( (up, up, up, down),      -s34 ),
                              ( (up, down, up, up),      s112 ),
                              ( (up, up, down, up),      s112 ),
                              ( (down, up, up, up),      s112 )]),
                # a=2 (T2, s₁=+1, s₂=-1/2)
                (-1//1, 2, [( (down, down, up, down),  -s23 ),
                              ( (down, up, down, down),  r6inv ),
                              ( (up, down, down, down),  r6inv )]),
                ( 0//1, 2, [( (down, down, up, up),     r3inv ),
                              ( (up, up, down, down),   -r3inv ),
                              ( (down, up, down, up),  -r12inv ),
                              ( (up, down, down, up),  -r12inv ),
                              ( (down, up, up, down),   r12inv ),
                              ( (up, down, up, down),   r12inv )]),
                ( 1//1, 2, [( (up, up, down, up),       s23 ),
                              ( (up, down, up, up),     -r6inv ),
                              ( (down, up, up, up),     -r6inv )]),
                # a=3 (T3, s₁=-1, s₂=+1/2)
                (-1//1, 3, [( (down, up, down, down),   r2inv ),
                              ( (up, down, down, down), -r2inv )]),
                ( 0//1, 3, [( (down, up, up, down),    -0.5 ),
                              ( (down, up, down, up),   -0.5 ),
                              ( (up, down, up, down),    0.5 ),
                              ( (up, down, down, up),    0.5 )]),
                ( 1//1, 3, [( (up, down, up, up),       r2inv ),
                              ( (down, up, up, up),     -r2inv )]),
            ])
        elseif J_r == 0//1 && irrep == "[2,2]"
            return _build_cg_coeffs(Val(4), j_r, J_r, irrep, multiplicity, dim_irrep, [
                # a=1 (T1, s₁=+1)
                (0//1, 1, [( (up, up, down, down),     -r3inv ),
                             ( (down, down, up, up),   -r3inv ),
                             ( (up, down, up, down),    r12inv ),
                             ( (up, down, down, up),    r12inv ),
                             ( (down, up, up, down),    r12inv ),
                             ( (down, up, down, up),    r12inv )]),
                # a=2 (T2, s₁=-1)
                (0//1, 2, [( (up, down, up, down),      0.5 ),
                             ( (up, down, down, up),    -0.5 ),
                             ( (down, up, up, down),    -0.5 ),
                             ( (down, up, down, up),     0.5 )]),
            ])
        else
            throw(ArgumentError("No such (J=$J_r, [κ]=$irrep) for N=4, j=1/2"))
        end

    elseif N == 2 && j_r == 1//1
        m0, m1, mm1 = 0//1, 1//1, -1//1
        r2inv = 1.0 / sqrt(2.0)
        r3inv = 1.0 / sqrt(3.0)
        r6inv = 1.0 / sqrt(6.0)

        if J_r == 2//1 && irrep == "[2]"
            return _build_cg_coeffs(Val(2), j_r, J_r, irrep, multiplicity, dim_irrep, [
                ( 2//1, 1, [((m1, m1), 1.0)]),
                ( 1//1, 1, [((m0, m1), r2inv), ((m1, m0), r2inv)]),
                ( 0//1, 1, [((m0, m0), sqrt(2.0/3.0)), ((m1, mm1), r6inv), ((mm1, m1), r6inv)]),
                (-1//1, 1, [((mm1, m0), r2inv), ((m0, mm1), r2inv)]),
                (-2//1, 1, [((mm1, mm1), 1.0)]),
            ])
        elseif J_r == 1//1 && irrep == "[1,1]"
            return _build_cg_coeffs(Val(2), j_r, J_r, irrep, multiplicity, dim_irrep, [
                ( 1//1, 1, [((m1, m0), -r2inv), ((m0, m1), r2inv)]),
                ( 0//1, 1, [((mm1, m1), -r2inv), ((m1, mm1), r2inv)]),
                (-1//1, 1, [((m0, mm1), r2inv), ((mm1, m0), -r2inv)]),
            ])
        elseif J_r == 0//1 && irrep == "[2]"
            return _build_cg_coeffs(Val(2), j_r, J_r, irrep, multiplicity, dim_irrep, [
                (0//1, 1, [((m1, mm1), -r3inv), ((m0, m0), r3inv), ((mm1, m1), -r3inv)]),
            ])
        else
            throw(ArgumentError("No such (J=$J_r, [κ]=$irrep) for N=2, j=1"))
        end

    elseif N == 3 && j_r == 1//1
        m0, m1, mm1 = 0//1, 1//1, -1//1
        r2inv = 1.0 / sqrt(2.0)
        r3inv = 1.0 / sqrt(3.0)
        r6inv = 1.0 / sqrt(6.0)
        r10inv = 1.0 / sqrt(10.0)
        r12inv = 1.0 / sqrt(12.0)
        r15inv = 1.0 / sqrt(15.0)
        s23 = sqrt(2.0/3.0)
        s25 = sqrt(2.0/5.0)
        s35 = sqrt(3.0/5.0)
        s415 = sqrt(4.0/15.0)

        if J_r == 3//1 && irrep == "[3]"
            # J=3, [3] totally symmetric, a=1 only
            return _build_cg_coeffs(Val(3), j_r, J_r, irrep, multiplicity, dim_irrep, [
                ( 3//1, 1, [((m1, m1, m1), -1.0)]),
                ( 2//1, 1, [((m0, m1, m1), r3inv), ((m1, m1, m0), r3inv), ((m1, m0, m1), r3inv)]),
                ( 1//1, 1, [((m1, m0, m0), s415), ((m0, m0, m1), s415), ((m0, m1, m0), s415),
                             ((m1, mm1, m1), r15inv), ((m1, m1, mm1), r15inv), ((mm1, m1, m1), r15inv)]),
                ( 0//1, 1, [((m0, m0, m0), -s25),
                             ((m1, m0, mm1), -r10inv), ((m0, m1, mm1), -r10inv), ((m1, mm1, m0), -r10inv),
                             ((mm1, m1, m0), -r10inv), ((m0, mm1, m1), -r10inv), ((mm1, m0, m1), -r10inv)]),
                (-1//1, 1, [((m0, mm1, m0), -s415), ((mm1, m0, m0), -s415), ((m0, m0, mm1), -s415),
                             ((mm1, m1, mm1), -r15inv), ((mm1, mm1, m1), -r15inv), ((m1, mm1, mm1), -r15inv)]),
                (-2//1, 1, [((m0, mm1, mm1), r3inv), ((mm1, m0, mm1), r3inv), ((mm1, mm1, m0), r3inv)]),
                (-3//1, 1, [((mm1, mm1, mm1), -1.0)]),
            ])
        elseif J_r == 2//1 && irrep == "[2,1]"
            # J=2, [2,1] mixed symmetry, a=1 (T1, s1=+1), a=2 (T2, s1=-1)
            return _build_cg_coeffs(Val(3), j_r, J_r, irrep, multiplicity, dim_irrep, [
                # a=1 (T1, s1 eigenvalue +1)
                ( 2//1, 1, [((m1, m1, m0), -s23), ((m1, m0, m1), r6inv), ((m0, m1, m1), r6inv)]),
                ( 1//1, 1, [((m0, m0, mm1), -r3inv), ((mm1, mm1, m1), r3inv),
                             ((mm1, m0, m0), r12inv), ((mm1, m1, mm1), -r12inv), ((m1, mm1, mm1), -r12inv), ((m0, mm1, m0), r12inv)]),
                ( 0//1, 1, [((m0, mm1, m1), -0.5), ((mm1, m0, m1), -0.5), ((m0, m1, mm1), 0.5), ((m1, m0, mm1), 0.5)]),
                (-1//1, 1, [((m1, m1, mm1), -r3inv), ((m0, m0, m1), r3inv),
                             ((mm1, m1, m1), r12inv), ((m1, mm1, m1), r12inv), ((m1, m0, m0), -r12inv), ((m0, m1, m0), -r12inv)]),
                (-2//1, 1, [((mm1, mm1, m0), -s23), ((m0, mm1, mm1), r6inv), ((mm1, m0, mm1), r6inv)]),
                # a=2 (T2, s1 eigenvalue -1)
                ( 2//1, 2, [((m1, m0, m1), r2inv), ((m0, m1, m1), -r2inv)]),
                ( 1//1, 2, [((m1, mm1, mm1), 0.5), ((m0, mm1, m0), 0.5), ((mm1, m1, mm1), -0.5), ((mm1, m0, m0), -0.5)]),
                ( 0//1, 2, [((mm1, m1, m0), -r3inv), ((m1, mm1, m0), r3inv),
                             ((m0, m1, mm1), -r12inv), ((mm1, m0, m1), -r12inv), ((m1, m0, mm1), r12inv), ((m0, mm1, m1), r12inv)]),
                (-1//1, 2, [((mm1, m1, m1), -0.5), ((m1, mm1, m1), 0.5), ((m0, m1, m0), -0.5), ((m1, m0, m0), 0.5)]),
                (-2//1, 2, [((m0, mm1, mm1), r2inv), ((mm1, m0, mm1), -r2inv)]),
            ])
        elseif J_r == 1//1 && irrep == "[3]"
            # J=1, [3] totally symmetric, a=1 only
            return _build_cg_coeffs(Val(3), j_r, J_r, irrep, multiplicity, dim_irrep, [
                ( 1//1, 1, [((m1, m1, mm1), -s415), ((m1, mm1, m1), -s415), ((mm1, m1, m1), -s415),
                             ((m0, m1, m0), r15inv), ((m0, m0, m1), r15inv), ((m1, m0, m0), r15inv)]),
                ( 0//1, 1, [((m0, m0, m0), s35),
                             ((m0, mm1, m1), -r15inv), ((m0, m1, mm1), -r15inv), ((mm1, m0, m1), -r15inv),
                             ((mm1, m1, m0), -r15inv), ((m1, mm1, m0), -r15inv), ((m1, m0, mm1), -r15inv)]),
                (-1//1, 1, [((mm1, mm1, m1), s415), ((m1, mm1, mm1), s415), ((mm1, m1, mm1), s415),
                             ((m0, mm1, m0), -r15inv), ((mm1, m0, m0), -r15inv), ((m0, m0, mm1), -r15inv)]),
            ])
        elseif J_r == 1//1 && irrep == "[2,1]"
            # J=1, [2,1] mixed symmetry, a=1 (T1), a=2 (T2)
            return _build_cg_coeffs(Val(3), j_r, J_r, irrep, multiplicity, dim_irrep, [
                # a=1 (T1, s1=+1)
                ( 1//1, 1, [((m0, m0, m1), r3inv), ((m1, m1, mm1), r3inv),
                             ((m1, m0, m0), -r12inv), ((m0, m1, m0), -r12inv), ((m1, mm1, m1), -r12inv), ((mm1, m1, m1), -r12inv)]),
                ( 0//1, 1, [((mm1, m1, m0), -r3inv), ((m1, mm1, m0), -r3inv),
                             ((mm1, m0, m1), r12inv), ((m0, mm1, m1), r12inv), ((m1, m0, mm1), r12inv), ((m0, m1, mm1), r12inv)]),
                (-1//1, 1, [((mm1, mm1, m1), r3inv), ((m0, m0, mm1), r3inv),
                             ((mm1, m0, m0), -r12inv), ((mm1, m1, mm1), -r12inv), ((m1, mm1, mm1), -r12inv), ((m0, mm1, m0), -r12inv)]),
                # a=2 (T2, s1=-1)
                ( 1//1, 2, [((m1, m0, m0), -0.5), ((m0, m1, m0), 0.5), ((m1, mm1, m1), 0.5), ((mm1, m1, m1), -0.5)]),
                ( 0//1, 2, [((m0, m1, mm1), 0.5), ((mm1, m0, m1), -0.5), ((m0, mm1, m1), 0.5), ((m1, m0, mm1), -0.5)]),
                (-1//1, 2, [((m1, mm1, mm1), 0.5), ((mm1, m1, mm1), -0.5), ((m0, mm1, m0), -0.5), ((mm1, m0, m0), 0.5)]),
            ])
        elseif J_r == 0//1 && irrep == "[1,1,1]"
            # J=0, [1,1,1] totally antisymmetric, a=1 only
            return _build_cg_coeffs(Val(3), j_r, J_r, irrep, multiplicity, dim_irrep, [
                (0//1, 1, [((mm1, m0, m1), r6inv), ((m1, m0, mm1), -r6inv), ((m0, mm1, m1), -r6inv),
                            ((m1, mm1, m0), r6inv), ((mm1, m1, m0), -r6inv), ((m0, m1, mm1), r6inv)]),
            ])
        else
            throw(ArgumentError("No such (J=$J_r, [κ]=$irrep) for N=3, j=1"))
        end

    elseif N == 4 && j_r == 1//1
        if J_r == 4//1 && irrep == "[4]"
            return _build_cg_coeffs(Val(4), j_r, J_r, irrep, multiplicity, dim_irrep, [
                (4//1, 1, [
                              ((1//1, 1//1, 1//1, 1//1), 1.0)]),
                (3//1, 1, [
                              ((0//1, 1//1, 1//1, 1//1), 0.5),
                              ((1//1, 1//1, 1//1, 0//1), 0.5),
                              ((1//1, 1//1, 0//1, 1//1), 0.5),
                              ((1//1, 0//1, 1//1, 1//1), 0.5)]),
                (2//1, 1, [
                              ((1//1, 1//1, 0//1, 0//1), -1.0 / sqrt(7.0)),
                              ((1//1, 0//1, 1//1, 0//1), -1.0 / sqrt(7.0)),
                              ((1//1, 0//1, 0//1, 1//1), -1.0 / sqrt(7.0)),
                              ((0//1, 0//1, 1//1, 1//1), -1.0 / sqrt(7.0)),
                              ((0//1, 1//1, 0//1, 1//1), -1.0 / sqrt(7.0)),
                              ((0//1, 1//1, 1//1, 0//1), -1.0 / sqrt(7.0)),
                              ((1//1, -1//1, 1//1, 1//1), -1.0 / sqrt(28.0)),
                              ((1//1, 1//1, 1//1, -1//1), -1.0 / sqrt(28.0)),
                              ((1//1, 1//1, -1//1, 1//1), -1.0 / sqrt(28.0)),
                              ((-1//1, 1//1, 1//1, 1//1), -1.0 / sqrt(28.0))]),
                (1//1, 1, [
                              ((0//1, 0//1, 1//1, 0//1), -1.0 / sqrt(7.0)),
                              ((0//1, 0//1, 0//1, 1//1), -1.0 / sqrt(7.0)),
                              ((1//1, 0//1, 0//1, 0//1), -1.0 / sqrt(7.0)),
                              ((0//1, 1//1, 0//1, 0//1), -1.0 / sqrt(7.0)),
                              ((0//1, -1//1, 1//1, 1//1), -1.0 / sqrt(28.0)),
                              ((-1//1, 1//1, 1//1, 0//1), -1.0 / sqrt(28.0)),
                              ((-1//1, 1//1, 0//1, 1//1), -1.0 / sqrt(28.0)),
                              ((1//1, 1//1, 0//1, -1//1), -1.0 / sqrt(28.0)),
                              ((1//1, 1//1, -1//1, 0//1), -1.0 / sqrt(28.0)),
                              ((1//1, 0//1, 1//1, -1//1), -1.0 / sqrt(28.0)),
                              ((1//1, 0//1, -1//1, 1//1), -1.0 / sqrt(28.0)),
                              ((1//1, -1//1, 1//1, 0//1), -1.0 / sqrt(28.0)),
                              ((1//1, -1//1, 0//1, 1//1), -1.0 / sqrt(28.0)),
                              ((0//1, 1//1, 1//1, -1//1), -1.0 / sqrt(28.0)),
                              ((0//1, 1//1, -1//1, 1//1), -1.0 / sqrt(28.0)),
                              ((-1//1, 0//1, 1//1, 1//1), -1.0 / sqrt(28.0))]),
                (0//1, 1, [
                              ((0//1, 0//1, 0//1, 0//1), -sqrt(8.0/35.0)),
                              ((0//1, 1//1, 0//1, -1//1), -sqrt(2.0/35.0)),
                              ((1//1, 0//1, -1//1, 0//1), -sqrt(2.0/35.0)),
                              ((0//1, -1//1, 1//1, 0//1), -sqrt(2.0/35.0)),
                              ((0//1, -1//1, 0//1, 1//1), -sqrt(2.0/35.0)),
                              ((1//1, 0//1, 0//1, -1//1), -sqrt(2.0/35.0)),
                              ((-1//1, 0//1, 1//1, 0//1), -sqrt(2.0/35.0)),
                              ((0//1, 1//1, -1//1, 0//1), -sqrt(2.0/35.0)),
                              ((-1//1, 0//1, 0//1, 1//1), -sqrt(2.0/35.0)),
                              ((1//1, -1//1, 0//1, 0//1), -sqrt(2.0/35.0)),
                              ((0//1, 0//1, 1//1, -1//1), -sqrt(2.0/35.0)),
                              ((0//1, 0//1, -1//1, 1//1), -sqrt(2.0/35.0)),
                              ((-1//1, 1//1, 0//1, 0//1), -sqrt(2.0/35.0)),
                              ((1//1, -1//1, 1//1, -1//1), -1.0 / sqrt(70.0)),
                              ((1//1, -1//1, -1//1, 1//1), -1.0 / sqrt(70.0)),
                              ((-1//1, 1//1, 1//1, -1//1), -1.0 / sqrt(70.0)),
                              ((-1//1, 1//1, -1//1, 1//1), -1.0 / sqrt(70.0)),
                              ((1//1, 1//1, -1//1, -1//1), -1.0 / sqrt(70.0)),
                              ((-1//1, -1//1, 1//1, 1//1), -1.0 / sqrt(70.0))]),
                (-1//1, 1, [
                              ((0//1, 0//1, 0//1, -1//1), -1.0 / sqrt(7.0)),
                              ((0//1, -1//1, 0//1, 0//1), -1.0 / sqrt(7.0)),
                              ((-1//1, 0//1, 0//1, 0//1), -1.0 / sqrt(7.0)),
                              ((0//1, 0//1, -1//1, 0//1), -1.0 / sqrt(7.0)),
                              ((0//1, 1//1, -1//1, -1//1), -1.0 / sqrt(28.0)),
                              ((1//1, -1//1, 0//1, -1//1), -1.0 / sqrt(28.0)),
                              ((1//1, 0//1, -1//1, -1//1), -1.0 / sqrt(28.0)),
                              ((-1//1, 1//1, 0//1, -1//1), -1.0 / sqrt(28.0)),
                              ((0//1, -1//1, 1//1, -1//1), -1.0 / sqrt(28.0)),
                              ((-1//1, 0//1, 1//1, -1//1), -1.0 / sqrt(28.0)),
                              ((-1//1, 1//1, -1//1, 0//1), -1.0 / sqrt(28.0)),
                              ((-1//1, -1//1, 1//1, 0//1), -1.0 / sqrt(28.0)),
                              ((0//1, -1//1, -1//1, 1//1), -1.0 / sqrt(28.0)),
                              ((-1//1, 0//1, -1//1, 1//1), -1.0 / sqrt(28.0)),
                              ((1//1, -1//1, -1//1, 0//1), -1.0 / sqrt(28.0)),
                              ((-1//1, -1//1, 0//1, 1//1), -1.0 / sqrt(28.0))]),
                (-2//1, 1, [
                              ((0//1, 0//1, -1//1, -1//1), -1.0 / sqrt(7.0)),
                              ((0//1, -1//1, -1//1, 0//1), -1.0 / sqrt(7.0)),
                              ((-1//1, 0//1, -1//1, 0//1), -1.0 / sqrt(7.0)),
                              ((-1//1, -1//1, 0//1, 0//1), -1.0 / sqrt(7.0)),
                              ((0//1, -1//1, 0//1, -1//1), -1.0 / sqrt(7.0)),
                              ((-1//1, 0//1, 0//1, -1//1), -1.0 / sqrt(7.0)),
                              ((-1//1, -1//1, -1//1, 1//1), -1.0 / sqrt(28.0)),
                              ((1//1, -1//1, -1//1, -1//1), -1.0 / sqrt(28.0)),
                              ((-1//1, 1//1, -1//1, -1//1), -1.0 / sqrt(28.0)),
                              ((-1//1, -1//1, 1//1, -1//1), -1.0 / sqrt(28.0))]),
                (-3//1, 1, [
                              ((-1//1, -1//1, -1//1, 0//1), -0.5),
                              ((0//1, -1//1, -1//1, -1//1), -0.5),
                              ((-1//1, 0//1, -1//1, -1//1), -0.5),
                              ((-1//1, -1//1, 0//1, -1//1), -0.5)]),
                (-4//1, 1, [
                              ((-1//1, -1//1, -1//1, -1//1), 1.0)])
            ])
        end

        if J_r == 3//1 && irrep == "[3,1]"
            return _build_cg_coeffs(Val(4), j_r, J_r, irrep, multiplicity, dim_irrep, [
                (3//1, 1, [
                              ((1//1, 1//1, 1//1, 0//1), -sqrt(3.0/4.0)),
                              ((0//1, 1//1, 1//1, 1//1), 1.0 / sqrt(12.0)),
                              ((1//1, 1//1, 0//1, 1//1), 1.0 / sqrt(12.0)),
                              ((1//1, 0//1, 1//1, 1//1), 1.0 / sqrt(12.0))]),
                (2//1, 1, [
                              ((1//1, 1//1, 1//1, -1//1), -0.5),
                              ((0//1, 1//1, 0//1, 1//1), 1.0 / sqrt(9.0)),
                              ((0//1, 1//1, 1//1, 0//1), -1.0 / sqrt(9.0)),
                              ((1//1, 1//1, 0//1, 0//1), -1.0 / sqrt(9.0)),
                              ((1//1, 0//1, 0//1, 1//1), 1.0 / sqrt(9.0)),
                              ((0//1, 0//1, 1//1, 1//1), 1.0 / sqrt(9.0)),
                              ((1//1, 0//1, 1//1, 0//1), -1.0 / sqrt(9.0)),
                              ((1//1, -1//1, 1//1, 1//1), 1.0 / sqrt(36.0)),
                              ((-1//1, 1//1, 1//1, 1//1), 1.0 / sqrt(36.0)),
                              ((1//1, 1//1, -1//1, 1//1), 1.0 / sqrt(36.0))]),
                (1//1, 1, [
                              ((0//1, 0//1, 0//1, 1//1), 1.0 / sqrt(5.0)),
                              ((1//1, 1//1, 0//1, -1//1), -sqrt(5.0/36.0)),
                              ((0//1, 1//1, 1//1, -1//1), -sqrt(5.0/36.0)),
                              ((1//1, 0//1, 1//1, -1//1), -sqrt(5.0/36.0)),
                              ((-1//1, 1//1, 0//1, 1//1), 1.0 / sqrt(20.0)),
                              ((1//1, 0//1, -1//1, 1//1), 1.0 / sqrt(20.0)),
                              ((-1//1, 0//1, 1//1, 1//1), 1.0 / sqrt(20.0)),
                              ((1//1, -1//1, 0//1, 1//1), 1.0 / sqrt(20.0)),
                              ((0//1, 1//1, -1//1, 1//1), 1.0 / sqrt(20.0)),
                              ((0//1, -1//1, 1//1, 1//1), 1.0 / sqrt(20.0)),
                              ((0//1, 0//1, 1//1, 0//1), -1.0 / sqrt(45.0)),
                              ((0//1, 1//1, 0//1, 0//1), -1.0 / sqrt(45.0)),
                              ((1//1, 0//1, 0//1, 0//1), -1.0 / sqrt(45.0)),
                              ((1//1, 1//1, -1//1, 0//1), -1.0 / sqrt(180.0)),
                              ((1//1, -1//1, 1//1, 0//1), -1.0 / sqrt(180.0)),
                              ((-1//1, 1//1, 1//1, 0//1), -1.0 / sqrt(180.0))]),
                (0//1, 1, [
                              ((1//1, 0//1, 0//1, -1//1), sqrt(2.0/15.0)),
                              ((0//1, 0//1, 1//1, -1//1), sqrt(2.0/15.0)),
                              ((0//1, 0//1, -1//1, 1//1), -sqrt(2.0/15.0)),
                              ((0//1, 1//1, 0//1, -1//1), sqrt(2.0/15.0)),
                              ((0//1, -1//1, 0//1, 1//1), -sqrt(2.0/15.0)),
                              ((-1//1, 0//1, 0//1, 1//1), -sqrt(2.0/15.0)),
                              ((1//1, -1//1, -1//1, 1//1), -1.0 / sqrt(30.0)),
                              ((1//1, -1//1, 1//1, -1//1), 1.0 / sqrt(30.0)),
                              ((1//1, 1//1, -1//1, -1//1), 1.0 / sqrt(30.0)),
                              ((-1//1, 1//1, 1//1, -1//1), 1.0 / sqrt(30.0)),
                              ((-1//1, 1//1, -1//1, 1//1), -1.0 / sqrt(30.0)),
                              ((-1//1, -1//1, 1//1, 1//1), -1.0 / sqrt(30.0))]),
                (-1//1, 1, [
                              ((0//1, 0//1, 0//1, -1//1), -1.0 / sqrt(5.0)),
                              ((-1//1, 0//1, -1//1, 1//1), sqrt(5.0/36.0)),
                              ((0//1, -1//1, -1//1, 1//1), sqrt(5.0/36.0)),
                              ((-1//1, -1//1, 0//1, 1//1), sqrt(5.0/36.0)),
                              ((1//1, 0//1, -1//1, -1//1), -1.0 / sqrt(20.0)),
                              ((-1//1, 0//1, 1//1, -1//1), -1.0 / sqrt(20.0)),
                              ((0//1, 1//1, -1//1, -1//1), -1.0 / sqrt(20.0)),
                              ((1//1, -1//1, 0//1, -1//1), -1.0 / sqrt(20.0)),
                              ((-1//1, 1//1, 0//1, -1//1), -1.0 / sqrt(20.0)),
                              ((0//1, -1//1, 1//1, -1//1), -1.0 / sqrt(20.0)),
                              ((0//1, -1//1, 0//1, 0//1), 1.0 / sqrt(45.0)),
                              ((0//1, 0//1, -1//1, 0//1), 1.0 / sqrt(45.0)),
                              ((-1//1, 0//1, 0//1, 0//1), 1.0 / sqrt(45.0)),
                              ((-1//1, -1//1, 1//1, 0//1), 1.0 / sqrt(180.0)),
                              ((-1//1, 1//1, -1//1, 0//1), 1.0 / sqrt(180.0)),
                              ((1//1, -1//1, -1//1, 0//1), 1.0 / sqrt(180.0))]),
                (-2//1, 1, [
                              ((-1//1, -1//1, -1//1, 1//1), -0.5),
                              ((0//1, 0//1, -1//1, -1//1), 1.0 / sqrt(9.0)),
                              ((0//1, -1//1, 0//1, -1//1), 1.0 / sqrt(9.0)),
                              ((0//1, -1//1, -1//1, 0//1), -1.0 / sqrt(9.0)),
                              ((-1//1, 0//1, -1//1, 0//1), -1.0 / sqrt(9.0)),
                              ((-1//1, -1//1, 0//1, 0//1), -1.0 / sqrt(9.0)),
                              ((-1//1, 0//1, 0//1, -1//1), 1.0 / sqrt(9.0)),
                              ((-1//1, -1//1, 1//1, -1//1), 1.0 / sqrt(36.0)),
                              ((1//1, -1//1, -1//1, -1//1), 1.0 / sqrt(36.0)),
                              ((-1//1, 1//1, -1//1, -1//1), 1.0 / sqrt(36.0))]),
                (-3//1, 1, [
                              ((-1//1, -1//1, -1//1, 0//1), sqrt(3.0/4.0)),
                              ((-1//1, 0//1, -1//1, -1//1), -1.0 / sqrt(12.0)),
                              ((-1//1, -1//1, 0//1, -1//1), -1.0 / sqrt(12.0)),
                              ((0//1, -1//1, -1//1, -1//1), -1.0 / sqrt(12.0))]),
                (3//1, 2, [
                              ((1//1, 1//1, 0//1, 1//1), sqrt(2.0/3.0)),
                              ((0//1, 1//1, 1//1, 1//1), -1.0 / sqrt(6.0)),
                              ((1//1, 0//1, 1//1, 1//1), -1.0 / sqrt(6.0))]),
                (2//1, 2, [
                              ((1//1, 1//1, 0//1, 0//1), -sqrt(2.0/9.0)),
                              ((1//1, 1//1, -1//1, 1//1), -sqrt(2.0/9.0)),
                              ((0//1, 0//1, 1//1, 1//1), sqrt(2.0/9.0)),
                              ((0//1, 1//1, 1//1, 0//1), 1.0 / sqrt(18.0)),
                              ((1//1, -1//1, 1//1, 1//1), 1.0 / sqrt(18.0)),
                              ((1//1, 0//1, 1//1, 0//1), 1.0 / sqrt(18.0)),
                              ((-1//1, 1//1, 1//1, 1//1), 1.0 / sqrt(18.0)),
                              ((1//1, 0//1, 0//1, 1//1), -1.0 / sqrt(18.0)),
                              ((0//1, 1//1, 0//1, 1//1), -1.0 / sqrt(18.0))]),
                (1//1, 2, [
                              ((1//1, 1//1, -1//1, 0//1), -sqrt(8.0/45.0)),
                              ((0//1, 0//1, 1//1, 0//1), sqrt(8.0/45.0)),
                              ((1//1, 0//1, -1//1, 1//1), -1.0 / sqrt(10.0)),
                              ((0//1, -1//1, 1//1, 1//1), 1.0 / sqrt(10.0)),
                              ((0//1, 1//1, -1//1, 1//1), -1.0 / sqrt(10.0)),
                              ((-1//1, 0//1, 1//1, 1//1), 1.0 / sqrt(10.0)),
                              ((1//1, 1//1, 0//1, -1//1), -sqrt(2.0/45.0)),
                              ((1//1, -1//1, 1//1, 0//1), sqrt(2.0/45.0)),
                              ((-1//1, 1//1, 1//1, 0//1), sqrt(2.0/45.0)),
                              ((1//1, 0//1, 0//1, 0//1), -sqrt(2.0/45.0)),
                              ((0//1, 1//1, 0//1, 0//1), -sqrt(2.0/45.0)),
                              ((1//1, 0//1, 1//1, -1//1), 1.0 / sqrt(90.0)),
                              ((0//1, 1//1, 1//1, -1//1), 1.0 / sqrt(90.0))]),
                (0//1, 2, [
                              ((0//1, -1//1, 1//1, 0//1), -sqrt(3.0/20.0)),
                              ((1//1, 0//1, -1//1, 0//1), sqrt(3.0/20.0)),
                              ((0//1, 1//1, -1//1, 0//1), sqrt(3.0/20.0)),
                              ((-1//1, 0//1, 1//1, 0//1), -sqrt(3.0/20.0)),
                              ((1//1, 1//1, -1//1, -1//1), 1.0 / sqrt(15.0)),
                              ((0//1, 0//1, 1//1, -1//1), -1.0 / sqrt(15.0)),
                              ((0//1, 0//1, -1//1, 1//1), 1.0 / sqrt(15.0)),
                              ((-1//1, -1//1, 1//1, 1//1), -1.0 / sqrt(15.0)),
                              ((1//1, -1//1, 1//1, -1//1), -sqrt(3.0/180.0)),
                              ((0//1, 1//1, 0//1, -1//1), sqrt(3.0/180.0)),
                              ((1//1, 0//1, 0//1, -1//1), sqrt(3.0/180.0)),
                              ((0//1, -1//1, 0//1, 1//1), -sqrt(3.0/180.0)),
                              ((-1//1, 1//1, 1//1, -1//1), -sqrt(3.0/180.0)),
                              ((-1//1, 0//1, 0//1, 1//1), -sqrt(3.0/180.0)),
                              ((1//1, -1//1, -1//1, 1//1), sqrt(3.0/180.0)),
                              ((-1//1, 1//1, -1//1, 1//1), sqrt(3.0/180.0))]),
                (-1//1, 2, [
                              ((0//1, 0//1, -1//1, 0//1), -sqrt(8.0/45.0)),
                              ((-1//1, -1//1, 1//1, 0//1), sqrt(8.0/45.0)),
                              ((0//1, 1//1, -1//1, -1//1), -1.0 / sqrt(10.0)),
                              ((1//1, 0//1, -1//1, -1//1), -1.0 / sqrt(10.0)),
                              ((-1//1, 0//1, 1//1, -1//1), 1.0 / sqrt(10.0)),
                              ((0//1, -1//1, 1//1, -1//1), 1.0 / sqrt(10.0)),
                              ((-1//1, 0//1, 0//1, 0//1), sqrt(2.0/45.0)),
                              ((0//1, -1//1, 0//1, 0//1), sqrt(2.0/45.0)),
                              ((-1//1, -1//1, 0//1, 1//1), sqrt(2.0/45.0)),
                              ((-1//1, 1//1, -1//1, 0//1), -sqrt(2.0/45.0)),
                              ((1//1, -1//1, -1//1, 0//1), -sqrt(2.0/45.0)),
                              ((-1//1, 0//1, -1//1, 1//1), -1.0 / sqrt(90.0)),
                              ((0//1, -1//1, -1//1, 1//1), -1.0 / sqrt(90.0))]),
                (-2//1, 2, [
                              ((-1//1, -1//1, 1//1, -1//1), sqrt(2.0/9.0)),
                              ((0//1, 0//1, -1//1, -1//1), -sqrt(2.0/9.0)),
                              ((-1//1, -1//1, 0//1, 0//1), sqrt(2.0/9.0)),
                              ((0//1, -1//1, -1//1, 0//1), -1.0 / sqrt(18.0)),
                              ((-1//1, 0//1, 0//1, -1//1), 1.0 / sqrt(18.0)),
                              ((0//1, -1//1, 0//1, -1//1), 1.0 / sqrt(18.0)),
                              ((-1//1, 1//1, -1//1, -1//1), -1.0 / sqrt(18.0)),
                              ((-1//1, 0//1, -1//1, 0//1), -1.0 / sqrt(18.0)),
                              ((1//1, -1//1, -1//1, -1//1), -1.0 / sqrt(18.0))]),
                (-3//1, 2, [
                              ((-1//1, -1//1, 0//1, -1//1), sqrt(2.0/3.0)),
                              ((0//1, -1//1, -1//1, -1//1), -1.0 / sqrt(6.0)),
                              ((-1//1, 0//1, -1//1, -1//1), -1.0 / sqrt(6.0))]),
                (3//1, 3, [
                              ((1//1, 0//1, 1//1, 1//1), -1.0 / sqrt(2.0)),
                              ((0//1, 1//1, 1//1, 1//1), 1.0 / sqrt(2.0))]),
                (2//1, 3, [
                              ((0//1, 1//1, 1//1, 0//1), 1.0 / sqrt(6.0)),
                              ((0//1, 1//1, 0//1, 1//1), 1.0 / sqrt(6.0)),
                              ((1//1, -1//1, 1//1, 1//1), -1.0 / sqrt(6.0)),
                              ((-1//1, 1//1, 1//1, 1//1), 1.0 / sqrt(6.0)),
                              ((1//1, 0//1, 1//1, 0//1), -1.0 / sqrt(6.0)),
                              ((1//1, 0//1, 0//1, 1//1), -1.0 / sqrt(6.0))]),
                (1//1, 3, [
                              ((1//1, 0//1, 0//1, 0//1), -sqrt(2.0/15.0)),
                              ((0//1, 1//1, 0//1, 0//1), sqrt(2.0/15.0)),
                              ((1//1, -1//1, 1//1, 0//1), -sqrt(2.0/15.0)),
                              ((-1//1, 1//1, 1//1, 0//1), sqrt(2.0/15.0)),
                              ((-1//1, 1//1, 0//1, 1//1), sqrt(2.0/15.0)),
                              ((1//1, -1//1, 0//1, 1//1), -sqrt(2.0/15.0)),
                              ((0//1, 1//1, 1//1, -1//1), 1.0 / sqrt(30.0)),
                              ((1//1, 0//1, 1//1, -1//1), -1.0 / sqrt(30.0)),
                              ((0//1, 1//1, -1//1, 1//1), 1.0 / sqrt(30.0)),
                              ((1//1, 0//1, -1//1, 1//1), -1.0 / sqrt(30.0)),
                              ((-1//1, 0//1, 1//1, 1//1), 1.0 / sqrt(30.0)),
                              ((0//1, -1//1, 1//1, 1//1), -1.0 / sqrt(30.0))]),
                (0//1, 3, [
                              ((1//1, -1//1, 0//1, 0//1), 1.0 / sqrt(5.0)),
                              ((-1//1, 1//1, 0//1, 0//1), -1.0 / sqrt(5.0)),
                              ((0//1, 1//1, 0//1, -1//1), -1.0 / sqrt(20.0)),
                              ((-1//1, 0//1, 1//1, 0//1), -1.0 / sqrt(20.0)),
                              ((0//1, 1//1, -1//1, 0//1), -1.0 / sqrt(20.0)),
                              ((0//1, -1//1, 0//1, 1//1), 1.0 / sqrt(20.0)),
                              ((0//1, -1//1, 1//1, 0//1), 1.0 / sqrt(20.0)),
                              ((1//1, 0//1, 0//1, -1//1), 1.0 / sqrt(20.0)),
                              ((-1//1, 1//1, 1//1, -1//1), -1.0 / sqrt(20.0)),
                              ((-1//1, 1//1, -1//1, 1//1), -1.0 / sqrt(20.0)),
                              ((-1//1, 0//1, 0//1, 1//1), -1.0 / sqrt(20.0)),
                              ((1//1, -1//1, 1//1, -1//1), 1.0 / sqrt(20.0)),
                              ((1//1, -1//1, -1//1, 1//1), 1.0 / sqrt(20.0)),
                              ((1//1, 0//1, -1//1, 0//1), 1.0 / sqrt(20.0))]),
                (-1//1, 3, [
                              ((-1//1, 0//1, 0//1, 0//1), -sqrt(2.0/15.0)),
                              ((-1//1, 1//1, 0//1, -1//1), -sqrt(2.0/15.0)),
                              ((0//1, -1//1, 0//1, 0//1), sqrt(2.0/15.0)),
                              ((1//1, -1//1, 0//1, -1//1), sqrt(2.0/15.0)),
                              ((1//1, -1//1, -1//1, 0//1), sqrt(2.0/15.0)),
                              ((-1//1, 1//1, -1//1, 0//1), -sqrt(2.0/15.0)),
                              ((-1//1, 0//1, 1//1, -1//1), -1.0 / sqrt(30.0)),
                              ((1//1, 0//1, -1//1, -1//1), 1.0 / sqrt(30.0)),
                              ((-1//1, 0//1, -1//1, 1//1), -1.0 / sqrt(30.0)),
                              ((0//1, -1//1, -1//1, 1//1), 1.0 / sqrt(30.0)),
                              ((0//1, 1//1, -1//1, -1//1), -1.0 / sqrt(30.0)),
                              ((0//1, -1//1, 1//1, -1//1), 1.0 / sqrt(30.0))]),
                (-2//1, 3, [
                              ((-1//1, 1//1, -1//1, -1//1), -1.0 / sqrt(6.0)),
                              ((-1//1, 0//1, -1//1, 0//1), -1.0 / sqrt(6.0)),
                              ((1//1, -1//1, -1//1, -1//1), 1.0 / sqrt(6.0)),
                              ((0//1, -1//1, 0//1, -1//1), 1.0 / sqrt(6.0)),
                              ((0//1, -1//1, -1//1, 0//1), 1.0 / sqrt(6.0)),
                              ((-1//1, 0//1, 0//1, -1//1), -1.0 / sqrt(6.0))]),
                (-3//1, 3, [
                              ((-1//1, 0//1, -1//1, -1//1), -1.0 / sqrt(2.0)),
                              ((0//1, -1//1, -1//1, -1//1), 1.0 / sqrt(2.0))])
            ])
        end

        if J_r == 2//1 && irrep == "[2,2]"
            return _build_cg_coeffs(Val(4), j_r, J_r, irrep, multiplicity, dim_irrep, [
                (2//1, 1, [
                              ((0//1, 0//1, 1//1, 1//1), -1.0 / sqrt(3.0)),
                              ((1//1, 1//1, 0//1, 0//1), -1.0 / sqrt(3.0)),
                              ((0//1, 1//1, 1//1, 0//1), 1.0 / sqrt(12.0)),
                              ((0//1, 1//1, 0//1, 1//1), 1.0 / sqrt(12.0)),
                              ((1//1, 0//1, 1//1, 0//1), 1.0 / sqrt(12.0)),
                              ((1//1, 0//1, 0//1, 1//1), 1.0 / sqrt(12.0))]),
                (1//1, 1, [
                              ((-1//1, 0//1, 1//1, 1//1), -1.0 / sqrt(6.0)),
                              ((0//1, -1//1, 1//1, 1//1), -1.0 / sqrt(6.0)),
                              ((1//1, 1//1, 0//1, -1//1), -1.0 / sqrt(6.0)),
                              ((1//1, 1//1, -1//1, 0//1), -1.0 / sqrt(6.0)),
                              ((1//1, 0//1, -1//1, 1//1), 1.0 / sqrt(24.0)),
                              ((0//1, 1//1, 1//1, -1//1), 1.0 / sqrt(24.0)),
                              ((0//1, 1//1, -1//1, 1//1), 1.0 / sqrt(24.0)),
                              ((1//1, 0//1, 1//1, -1//1), 1.0 / sqrt(24.0)),
                              ((-1//1, 1//1, 0//1, 1//1), 1.0 / sqrt(24.0)),
                              ((-1//1, 1//1, 1//1, 0//1), 1.0 / sqrt(24.0)),
                              ((1//1, -1//1, 1//1, 0//1), 1.0 / sqrt(24.0)),
                              ((1//1, -1//1, 0//1, 1//1), 1.0 / sqrt(24.0))]),
                (0//1, 1, [
                              ((1//1, 1//1, -1//1, -1//1), -sqrt(2.0/9.0)),
                              ((-1//1, -1//1, 1//1, 1//1), -sqrt(2.0/9.0)),
                              ((0//1, 0//1, -1//1, 1//1), 1.0 / sqrt(18.0)),
                              ((0//1, 0//1, 1//1, -1//1), 1.0 / sqrt(18.0)),
                              ((1//1, -1//1, -1//1, 1//1), 1.0 / sqrt(18.0)),
                              ((-1//1, 1//1, 1//1, -1//1), 1.0 / sqrt(18.0)),
                              ((1//1, -1//1, 1//1, -1//1), 1.0 / sqrt(18.0)),
                              ((-1//1, 1//1, -1//1, 1//1), 1.0 / sqrt(18.0)),
                              ((1//1, -1//1, 0//1, 0//1), 1.0 / sqrt(18.0)),
                              ((-1//1, 1//1, 0//1, 0//1), 1.0 / sqrt(18.0)),
                              ((-1//1, 0//1, 0//1, 1//1), -1.0 / sqrt(72.0)),
                              ((-1//1, 0//1, 1//1, 0//1), -1.0 / sqrt(72.0)),
                              ((1//1, 0//1, -1//1, 0//1), -1.0 / sqrt(72.0)),
                              ((0//1, -1//1, 1//1, 0//1), -1.0 / sqrt(72.0)),
                              ((1//1, 0//1, 0//1, -1//1), -1.0 / sqrt(72.0)),
                              ((0//1, 1//1, 0//1, -1//1), -1.0 / sqrt(72.0)),
                              ((0//1, 1//1, -1//1, 0//1), -1.0 / sqrt(72.0)),
                              ((0//1, -1//1, 0//1, 1//1), -1.0 / sqrt(72.0))]),
                (-1//1, 1, [
                              ((1//1, 0//1, -1//1, -1//1), -1.0 / sqrt(6.0)),
                              ((-1//1, -1//1, 1//1, 0//1), -1.0 / sqrt(6.0)),
                              ((-1//1, -1//1, 0//1, 1//1), -1.0 / sqrt(6.0)),
                              ((0//1, 1//1, -1//1, -1//1), -1.0 / sqrt(6.0)),
                              ((0//1, -1//1, -1//1, 1//1), 1.0 / sqrt(24.0)),
                              ((0//1, -1//1, 1//1, -1//1), 1.0 / sqrt(24.0)),
                              ((1//1, -1//1, 0//1, -1//1), 1.0 / sqrt(24.0)),
                              ((-1//1, 1//1, 0//1, -1//1), 1.0 / sqrt(24.0)),
                              ((-1//1, 1//1, -1//1, 0//1), 1.0 / sqrt(24.0)),
                              ((-1//1, 0//1, -1//1, 1//1), 1.0 / sqrt(24.0)),
                              ((1//1, -1//1, -1//1, 0//1), 1.0 / sqrt(24.0)),
                              ((-1//1, 0//1, 1//1, -1//1), 1.0 / sqrt(24.0))]),
                (-2//1, 1, [
                              ((0//1, 0//1, -1//1, -1//1), 1.0 / sqrt(3.0)),
                              ((-1//1, -1//1, 0//1, 0//1), 1.0 / sqrt(3.0)),
                              ((0//1, -1//1, -1//1, 0//1), -1.0 / sqrt(12.0)),
                              ((0//1, -1//1, 0//1, -1//1), -1.0 / sqrt(12.0)),
                              ((-1//1, 0//1, -1//1, 0//1), -1.0 / sqrt(12.0)),
                              ((-1//1, 0//1, 0//1, -1//1), -1.0 / sqrt(12.0))]),
                (2//1, 2, [
                              ((1//1, 0//1, 1//1, 0//1), -0.5),
                              ((1//1, 0//1, 0//1, 1//1), 0.5),
                              ((0//1, 1//1, 0//1, 1//1), -0.5),
                              ((0//1, 1//1, 1//1, 0//1), 0.5)]),
                (1//1, 2, [
                              ((-1//1, 1//1, 0//1, 1//1), 1.0 / sqrt(8.0)),
                              ((1//1, 0//1, 1//1, -1//1), 1.0 / sqrt(8.0)),
                              ((-1//1, 1//1, 1//1, 0//1), -1.0 / sqrt(8.0)),
                              ((0//1, 1//1, -1//1, 1//1), 1.0 / sqrt(8.0)),
                              ((1//1, -1//1, 0//1, 1//1), -1.0 / sqrt(8.0)),
                              ((1//1, 0//1, -1//1, 1//1), -1.0 / sqrt(8.0)),
                              ((0//1, 1//1, 1//1, -1//1), -1.0 / sqrt(8.0)),
                              ((1//1, -1//1, 1//1, 0//1), 1.0 / sqrt(8.0))]),
                (0//1, 2, [
                              ((-1//1, 1//1, -1//1, 1//1), 1.0 / sqrt(6.0)),
                              ((1//1, -1//1, 1//1, -1//1), 1.0 / sqrt(6.0)),
                              ((-1//1, 1//1, 1//1, -1//1), -1.0 / sqrt(6.0)),
                              ((1//1, -1//1, -1//1, 1//1), -1.0 / sqrt(6.0)),
                              ((0//1, -1//1, 0//1, 1//1), -1.0 / sqrt(24.0)),
                              ((0//1, 1//1, 0//1, -1//1), -1.0 / sqrt(24.0)),
                              ((0//1, 1//1, -1//1, 0//1), 1.0 / sqrt(24.0)),
                              ((0//1, -1//1, 1//1, 0//1), 1.0 / sqrt(24.0)),
                              ((-1//1, 0//1, 0//1, 1//1), 1.0 / sqrt(24.0)),
                              ((1//1, 0//1, 0//1, -1//1), 1.0 / sqrt(24.0)),
                              ((1//1, 0//1, -1//1, 0//1), -1.0 / sqrt(24.0)),
                              ((-1//1, 0//1, 1//1, 0//1), -1.0 / sqrt(24.0))]),
                (-1//1, 2, [
                              ((-1//1, 0//1, -1//1, 1//1), 1.0 / sqrt(8.0)),
                              ((0//1, -1//1, 1//1, -1//1), 1.0 / sqrt(8.0)),
                              ((1//1, -1//1, 0//1, -1//1), 1.0 / sqrt(8.0)),
                              ((-1//1, 1//1, 0//1, -1//1), -1.0 / sqrt(8.0)),
                              ((-1//1, 1//1, -1//1, 0//1), 1.0 / sqrt(8.0)),
                              ((1//1, -1//1, -1//1, 0//1), -1.0 / sqrt(8.0)),
                              ((0//1, -1//1, -1//1, 1//1), -1.0 / sqrt(8.0)),
                              ((-1//1, 0//1, 1//1, -1//1), -1.0 / sqrt(8.0))]),
                (-2//1, 2, [
                              ((-1//1, 0//1, -1//1, 0//1), 0.5),
                              ((0//1, -1//1, 0//1, -1//1), 0.5),
                              ((-1//1, 0//1, 0//1, -1//1), -0.5),
                              ((0//1, -1//1, -1//1, 0//1), -0.5)])
            ])
        end

        if J_r == 2//1 && irrep == "[3,1]"
            return _build_cg_coeffs(Val(4), j_r, J_r, irrep, multiplicity, dim_irrep, [
                (2//1, 1, [
                              ((1//1, 1//1, 1//1, -1//1), -1.0 / sqrt(2.0)),
                              ((1//1, 1//1, -1//1, 1//1), 1.0 / sqrt(18.0)),
                              ((1//1, 1//1, 0//1, 0//1), 1.0 / sqrt(18.0)),
                              ((-1//1, 1//1, 1//1, 1//1), 1.0 / sqrt(18.0)),
                              ((0//1, 1//1, 1//1, 0//1), 1.0 / sqrt(18.0)),
                              ((1//1, -1//1, 1//1, 1//1), 1.0 / sqrt(18.0)),
                              ((1//1, 0//1, 0//1, 1//1), -1.0 / sqrt(18.0)),
                              ((1//1, 0//1, 1//1, 0//1), 1.0 / sqrt(18.0)),
                              ((0//1, 1//1, 0//1, 1//1), -1.0 / sqrt(18.0)),
                              ((0//1, 0//1, 1//1, 1//1), -1.0 / sqrt(18.0))]),
                (1//1, 1, [
                              ((0//1, 0//1, 0//1, 1//1), -0.5),
                              ((1//1, 1//1, -1//1, 0//1), 1.0 / sqrt(9.0)),
                              ((1//1, 0//1, 1//1, -1//1), -1.0 / sqrt(9.0)),
                              ((0//1, 1//1, 1//1, -1//1), -1.0 / sqrt(9.0)),
                              ((-1//1, 1//1, 1//1, 0//1), 1.0 / sqrt(9.0)),
                              ((1//1, 1//1, 0//1, -1//1), -1.0 / sqrt(9.0)),
                              ((1//1, -1//1, 1//1, 0//1), 1.0 / sqrt(9.0)),
                              ((0//1, 1//1, 0//1, 0//1), 1.0 / sqrt(36.0)),
                              ((0//1, 0//1, 1//1, 0//1), 1.0 / sqrt(36.0)),
                              ((1//1, 0//1, 0//1, 0//1), 1.0 / sqrt(36.0))]),
                (0//1, 1, [
                              ((0//1, 0//1, -1//1, 1//1), -1.0 / sqrt(12.0)),
                              ((0//1, 1//1, 0//1, -1//1), -1.0 / sqrt(12.0)),
                              ((0//1, -1//1, 1//1, 0//1), 1.0 / sqrt(12.0)),
                              ((0//1, 1//1, -1//1, 0//1), 1.0 / sqrt(12.0)),
                              ((-1//1, 0//1, 1//1, 0//1), 1.0 / sqrt(12.0)),
                              ((-1//1, 0//1, 0//1, 1//1), -1.0 / sqrt(12.0)),
                              ((1//1, -1//1, 0//1, 0//1), 1.0 / sqrt(12.0)),
                              ((1//1, 0//1, -1//1, 0//1), 1.0 / sqrt(12.0)),
                              ((0//1, 0//1, 1//1, -1//1), -1.0 / sqrt(12.0)),
                              ((0//1, -1//1, 0//1, 1//1), -1.0 / sqrt(12.0)),
                              ((-1//1, 1//1, 0//1, 0//1), 1.0 / sqrt(12.0)),
                              ((1//1, 0//1, 0//1, -1//1), -1.0 / sqrt(12.0))]),
                (-1//1, 1, [
                              ((0//1, 0//1, 0//1, -1//1), -0.5),
                              ((-1//1, 0//1, -1//1, 1//1), -1.0 / sqrt(9.0)),
                              ((0//1, -1//1, -1//1, 1//1), -1.0 / sqrt(9.0)),
                              ((-1//1, -1//1, 0//1, 1//1), -1.0 / sqrt(9.0)),
                              ((1//1, -1//1, -1//1, 0//1), 1.0 / sqrt(9.0)),
                              ((-1//1, 1//1, -1//1, 0//1), 1.0 / sqrt(9.0)),
                              ((-1//1, -1//1, 1//1, 0//1), 1.0 / sqrt(9.0)),
                              ((-1//1, 0//1, 0//1, 0//1), 1.0 / sqrt(36.0)),
                              ((0//1, -1//1, 0//1, 0//1), 1.0 / sqrt(36.0)),
                              ((0//1, 0//1, -1//1, 0//1), 1.0 / sqrt(36.0))]),
                (-2//1, 1, [
                              ((-1//1, -1//1, -1//1, 1//1), -1.0 / sqrt(2.0)),
                              ((-1//1, 0//1, 0//1, -1//1), -1.0 / sqrt(18.0)),
                              ((-1//1, 0//1, -1//1, 0//1), 1.0 / sqrt(18.0)),
                              ((1//1, -1//1, -1//1, -1//1), 1.0 / sqrt(18.0)),
                              ((-1//1, 1//1, -1//1, -1//1), 1.0 / sqrt(18.0)),
                              ((0//1, 0//1, -1//1, -1//1), -1.0 / sqrt(18.0)),
                              ((-1//1, -1//1, 0//1, 0//1), 1.0 / sqrt(18.0)),
                              ((0//1, -1//1, -1//1, 0//1), 1.0 / sqrt(18.0)),
                              ((-1//1, -1//1, 1//1, -1//1), 1.0 / sqrt(18.0)),
                              ((0//1, -1//1, 0//1, -1//1), -1.0 / sqrt(18.0))]),
                (2//1, 2, [
                              ((1//1, 1//1, -1//1, 1//1), -sqrt(4.0/9.0)),
                              ((1//1, 1//1, 0//1, 0//1), 1.0 / sqrt(9.0)),
                              ((-1//1, 1//1, 1//1, 1//1), 1.0 / sqrt(9.0)),
                              ((0//1, 0//1, 1//1, 1//1), -1.0 / sqrt(9.0)),
                              ((1//1, -1//1, 1//1, 1//1), 1.0 / sqrt(9.0)),
                              ((1//1, 0//1, 1//1, 0//1), -1.0 / sqrt(36.0)),
                              ((0//1, 1//1, 0//1, 1//1), 1.0 / sqrt(36.0)),
                              ((1//1, 0//1, 0//1, 1//1), 1.0 / sqrt(36.0)),
                              ((0//1, 1//1, 1//1, 0//1), -1.0 / sqrt(36.0))]),
                (1//1, 2, [
                              ((0//1, 0//1, 1//1, 0//1), sqrt(2.0/9.0)),
                              ((0//1, 1//1, -1//1, 1//1), 1.0 / sqrt(8.0)),
                              ((-1//1, 1//1, 0//1, 1//1), -1.0 / sqrt(8.0)),
                              ((1//1, 0//1, -1//1, 1//1), 1.0 / sqrt(8.0)),
                              ((1//1, -1//1, 0//1, 1//1), -1.0 / sqrt(8.0)),
                              ((1//1, 1//1, 0//1, -1//1), -1.0 / sqrt(18.0)),
                              ((1//1, 0//1, 0//1, 0//1), -1.0 / sqrt(18.0)),
                              ((0//1, 1//1, 0//1, 0//1), -1.0 / sqrt(18.0)),
                              ((1//1, 1//1, -1//1, 0//1), 1.0 / sqrt(18.0)),
                              ((0//1, 1//1, 1//1, -1//1), 1.0 / sqrt(72.0)),
                              ((1//1, 0//1, 1//1, -1//1), 1.0 / sqrt(72.0)),
                              ((1//1, -1//1, 1//1, 0//1), -1.0 / sqrt(72.0)),
                              ((-1//1, 1//1, 1//1, 0//1), -1.0 / sqrt(72.0))]),
                (0//1, 2, [
                              ((0//1, 0//1, -1//1, 1//1), 1.0 / sqrt(6.0)),
                              ((0//1, 0//1, 1//1, -1//1), 1.0 / sqrt(6.0)),
                              ((-1//1, 1//1, 0//1, 0//1), -1.0 / sqrt(6.0)),
                              ((1//1, -1//1, 0//1, 0//1), -1.0 / sqrt(6.0)),
                              ((0//1, -1//1, 1//1, 0//1), 1.0 / sqrt(24.0)),
                              ((0//1, 1//1, 0//1, -1//1), -1.0 / sqrt(24.0)),
                              ((1//1, 0//1, -1//1, 0//1), 1.0 / sqrt(24.0)),
                              ((0//1, -1//1, 0//1, 1//1), -1.0 / sqrt(24.0)),
                              ((-1//1, 0//1, 0//1, 1//1), -1.0 / sqrt(24.0)),
                              ((-1//1, 0//1, 1//1, 0//1), 1.0 / sqrt(24.0)),
                              ((0//1, 1//1, -1//1, 0//1), 1.0 / sqrt(24.0)),
                              ((1//1, 0//1, 0//1, -1//1), -1.0 / sqrt(24.0))]),
                (-1//1, 2, [
                              ((0//1, 0//1, -1//1, 0//1), sqrt(2.0/9.0)),
                              ((0//1, -1//1, 1//1, -1//1), 1.0 / sqrt(8.0)),
                              ((-1//1, 0//1, 1//1, -1//1), 1.0 / sqrt(8.0)),
                              ((-1//1, 1//1, 0//1, -1//1), -1.0 / sqrt(8.0)),
                              ((1//1, -1//1, 0//1, -1//1), -1.0 / sqrt(8.0)),
                              ((0//1, -1//1, 0//1, 0//1), -1.0 / sqrt(18.0)),
                              ((-1//1, 0//1, 0//1, 0//1), -1.0 / sqrt(18.0)),
                              ((-1//1, -1//1, 1//1, 0//1), 1.0 / sqrt(18.0)),
                              ((-1//1, -1//1, 0//1, 1//1), -1.0 / sqrt(18.0)),
                              ((-1//1, 0//1, -1//1, 1//1), 1.0 / sqrt(72.0)),
                              ((0//1, -1//1, -1//1, 1//1), 1.0 / sqrt(72.0)),
                              ((1//1, -1//1, -1//1, 0//1), -1.0 / sqrt(72.0)),
                              ((-1//1, 1//1, -1//1, 0//1), -1.0 / sqrt(72.0))]),
                (-2//1, 2, [
                              ((-1//1, -1//1, 1//1, -1//1), sqrt(4.0/9.0)),
                              ((0//1, 0//1, -1//1, -1//1), 1.0 / sqrt(9.0)),
                              ((-1//1, -1//1, 0//1, 0//1), -1.0 / sqrt(9.0)),
                              ((-1//1, 1//1, -1//1, -1//1), -1.0 / sqrt(9.0)),
                              ((1//1, -1//1, -1//1, -1//1), -1.0 / sqrt(9.0)),
                              ((-1//1, 0//1, 0//1, -1//1), -1.0 / sqrt(36.0)),
                              ((0//1, -1//1, 0//1, -1//1), -1.0 / sqrt(36.0)),
                              ((-1//1, 0//1, -1//1, 0//1), 1.0 / sqrt(36.0)),
                              ((0//1, -1//1, -1//1, 0//1), 1.0 / sqrt(36.0))]),
                (2//1, 3, [
                              ((-1//1, 1//1, 1//1, 1//1), -1.0 / sqrt(3.0)),
                              ((1//1, -1//1, 1//1, 1//1), 1.0 / sqrt(3.0)),
                              ((1//1, 0//1, 1//1, 0//1), -1.0 / sqrt(12.0)),
                              ((1//1, 0//1, 0//1, 1//1), -1.0 / sqrt(12.0)),
                              ((0//1, 1//1, 1//1, 0//1), 1.0 / sqrt(12.0)),
                              ((0//1, 1//1, 0//1, 1//1), 1.0 / sqrt(12.0))]),
                (1//1, 3, [
                              ((-1//1, 0//1, 1//1, 1//1), -1.0 / sqrt(6.0)),
                              ((1//1, 0//1, 0//1, 0//1), -1.0 / sqrt(6.0)),
                              ((0//1, -1//1, 1//1, 1//1), 1.0 / sqrt(6.0)),
                              ((0//1, 1//1, 0//1, 0//1), 1.0 / sqrt(6.0)),
                              ((1//1, -1//1, 1//1, 0//1), 1.0 / sqrt(24.0)),
                              ((1//1, 0//1, 1//1, -1//1), -1.0 / sqrt(24.0)),
                              ((1//1, -1//1, 0//1, 1//1), 1.0 / sqrt(24.0)),
                              ((-1//1, 1//1, 1//1, 0//1), -1.0 / sqrt(24.0)),
                              ((0//1, 1//1, 1//1, -1//1), 1.0 / sqrt(24.0)),
                              ((-1//1, 1//1, 0//1, 1//1), -1.0 / sqrt(24.0)),
                              ((0//1, 1//1, -1//1, 1//1), 1.0 / sqrt(24.0)),
                              ((1//1, 0//1, -1//1, 1//1), -1.0 / sqrt(24.0))]),
                (0//1, 3, [
                              ((-1//1, 0//1, 1//1, 0//1), -1.0 / sqrt(8.0)),
                              ((0//1, 1//1, 0//1, -1//1), 1.0 / sqrt(8.0)),
                              ((-1//1, 0//1, 0//1, 1//1), -1.0 / sqrt(8.0)),
                              ((1//1, 0//1, -1//1, 0//1), -1.0 / sqrt(8.0)),
                              ((1//1, 0//1, 0//1, -1//1), -1.0 / sqrt(8.0)),
                              ((0//1, 1//1, -1//1, 0//1), 1.0 / sqrt(8.0)),
                              ((0//1, -1//1, 0//1, 1//1), 1.0 / sqrt(8.0)),
                              ((0//1, -1//1, 1//1, 0//1), 1.0 / sqrt(8.0))]),
                (-1//1, 3, [
                              ((1//1, 0//1, -1//1, -1//1), 1.0 / sqrt(6.0)),
                              ((0//1, -1//1, 0//1, 0//1), -1.0 / sqrt(6.0)),
                              ((0//1, 1//1, -1//1, -1//1), -1.0 / sqrt(6.0)),
                              ((-1//1, 0//1, 0//1, 0//1), 1.0 / sqrt(6.0)),
                              ((1//1, -1//1, 0//1, -1//1), 1.0 / sqrt(24.0)),
                              ((-1//1, 0//1, -1//1, 1//1), 1.0 / sqrt(24.0)),
                              ((1//1, -1//1, -1//1, 0//1), 1.0 / sqrt(24.0)),
                              ((-1//1, 0//1, 1//1, -1//1), 1.0 / sqrt(24.0)),
                              ((0//1, -1//1, -1//1, 1//1), -1.0 / sqrt(24.0)),
                              ((-1//1, 1//1, 0//1, -1//1), -1.0 / sqrt(24.0)),
                              ((-1//1, 1//1, -1//1, 0//1), -1.0 / sqrt(24.0)),
                              ((0//1, -1//1, 1//1, -1//1), -1.0 / sqrt(24.0))]),
                (-2//1, 3, [
                              ((-1//1, 1//1, -1//1, -1//1), -1.0 / sqrt(3.0)),
                              ((1//1, -1//1, -1//1, -1//1), 1.0 / sqrt(3.0)),
                              ((0//1, -1//1, 0//1, -1//1), -1.0 / sqrt(12.0)),
                              ((-1//1, 0//1, 0//1, -1//1), 1.0 / sqrt(12.0)),
                              ((-1//1, 0//1, -1//1, 0//1), 1.0 / sqrt(12.0)),
                              ((0//1, -1//1, -1//1, 0//1), -1.0 / sqrt(12.0))])
            ])
        end

        if J_r == 2//1 && irrep == "[4]"
            return _build_cg_coeffs(Val(4), j_r, J_r, irrep, multiplicity, dim_irrep, [
                (2//1, 1, [
                              ((-1//1, 1//1, 1//1, 1//1), sqrt(3.0/14.0)),
                              ((1//1, 1//1, 1//1, -1//1), sqrt(3.0/14.0)),
                              ((1//1, -1//1, 1//1, 1//1), sqrt(3.0/14.0)),
                              ((1//1, 1//1, -1//1, 1//1), sqrt(3.0/14.0)),
                              ((1//1, 0//1, 1//1, 0//1), -1.0 / sqrt(42.0)),
                              ((1//1, 1//1, 0//1, 0//1), -1.0 / sqrt(42.0)),
                              ((0//1, 1//1, 1//1, 0//1), -1.0 / sqrt(42.0)),
                              ((0//1, 0//1, 1//1, 1//1), -1.0 / sqrt(42.0)),
                              ((1//1, 0//1, 0//1, 1//1), -1.0 / sqrt(42.0)),
                              ((0//1, 1//1, 0//1, 1//1), -1.0 / sqrt(42.0))]),
                (1//1, 1, [
                              ((1//1, 0//1, 0//1, 0//1), sqrt(3.0/28.0)),
                              ((0//1, 1//1, 0//1, 0//1), sqrt(3.0/28.0)),
                              ((0//1, 0//1, 1//1, 0//1), sqrt(3.0/28.0)),
                              ((0//1, 0//1, 0//1, 1//1), sqrt(3.0/28.0)),
                              ((1//1, -1//1, 1//1, 0//1), -sqrt(2.0/42.0)),
                              ((1//1, -1//1, 0//1, 1//1), -sqrt(2.0/42.0)),
                              ((0//1, -1//1, 1//1, 1//1), -sqrt(2.0/42.0)),
                              ((-1//1, 1//1, 1//1, 0//1), -sqrt(2.0/42.0)),
                              ((0//1, 1//1, -1//1, 1//1), -sqrt(2.0/42.0)),
                              ((-1//1, 0//1, 1//1, 1//1), -sqrt(2.0/42.0)),
                              ((0//1, 1//1, 1//1, -1//1), -sqrt(2.0/42.0)),
                              ((1//1, 0//1, -1//1, 1//1), -sqrt(2.0/42.0)),
                              ((-1//1, 1//1, 0//1, 1//1), -sqrt(2.0/42.0)),
                              ((1//1, 0//1, 1//1, -1//1), -sqrt(2.0/42.0)),
                              ((1//1, 1//1, -1//1, 0//1), -sqrt(2.0/42.0)),
                              ((1//1, 1//1, 0//1, -1//1), -sqrt(2.0/42.0))]),
                (0//1, 1, [
                              ((0//1, 0//1, 0//1, 0//1), -sqrt(4.0/7.0)),
                              ((1//1, -1//1, 1//1, -1//1), sqrt(4.0/63.0)),
                              ((-1//1, 1//1, 1//1, -1//1), sqrt(4.0/63.0)),
                              ((1//1, 1//1, -1//1, -1//1), sqrt(4.0/63.0)),
                              ((1//1, -1//1, -1//1, 1//1), sqrt(4.0/63.0)),
                              ((-1//1, 1//1, -1//1, 1//1), sqrt(4.0/63.0)),
                              ((-1//1, -1//1, 1//1, 1//1), sqrt(4.0/63.0)),
                              ((1//1, -1//1, 0//1, 0//1), 1.0 / sqrt(252.0)),
                              ((0//1, 0//1, -1//1, 1//1), 1.0 / sqrt(252.0)),
                              ((0//1, -1//1, 1//1, 0//1), 1.0 / sqrt(252.0)),
                              ((-1//1, 1//1, 0//1, 0//1), 1.0 / sqrt(252.0)),
                              ((0//1, 1//1, -1//1, 0//1), 1.0 / sqrt(252.0)),
                              ((-1//1, 0//1, 0//1, 1//1), 1.0 / sqrt(252.0)),
                              ((0//1, -1//1, 0//1, 1//1), 1.0 / sqrt(252.0)),
                              ((0//1, 1//1, 0//1, -1//1), 1.0 / sqrt(252.0)),
                              ((0//1, 0//1, 1//1, -1//1), 1.0 / sqrt(252.0)),
                              ((1//1, 0//1, -1//1, 0//1), 1.0 / sqrt(252.0)),
                              ((1//1, 0//1, 0//1, -1//1), 1.0 / sqrt(252.0)),
                              ((-1//1, 0//1, 1//1, 0//1), 1.0 / sqrt(252.0))]),
                (-1//1, 1, [
                              ((0//1, -1//1, 0//1, 0//1), sqrt(3.0/28.0)),
                              ((0//1, 0//1, -1//1, 0//1), sqrt(3.0/28.0)),
                              ((-1//1, 0//1, 0//1, 0//1), sqrt(3.0/28.0)),
                              ((0//1, 0//1, 0//1, -1//1), sqrt(3.0/28.0)),
                              ((-1//1, 0//1, -1//1, 1//1), -sqrt(2.0/42.0)),
                              ((1//1, 0//1, -1//1, -1//1), -sqrt(2.0/42.0)),
                              ((-1//1, 0//1, 1//1, -1//1), -sqrt(2.0/42.0)),
                              ((0//1, -1//1, 1//1, -1//1), -sqrt(2.0/42.0)),
                              ((-1//1, -1//1, 0//1, 1//1), -sqrt(2.0/42.0)),
                              ((1//1, -1//1, 0//1, -1//1), -sqrt(2.0/42.0)),
                              ((0//1, -1//1, -1//1, 1//1), -sqrt(2.0/42.0)),
                              ((-1//1, -1//1, 1//1, 0//1), -sqrt(2.0/42.0)),
                              ((1//1, -1//1, -1//1, 0//1), -sqrt(2.0/42.0)),
                              ((0//1, 1//1, -1//1, -1//1), -sqrt(2.0/42.0)),
                              ((-1//1, 1//1, -1//1, 0//1), -sqrt(2.0/42.0)),
                              ((-1//1, 1//1, 0//1, -1//1), -sqrt(2.0/42.0))]),
                (-2//1, 1, [
                              ((-1//1, 1//1, -1//1, -1//1), -sqrt(3.0/14.0)),
                              ((-1//1, -1//1, -1//1, 1//1), -sqrt(3.0/14.0)),
                              ((-1//1, -1//1, 1//1, -1//1), -sqrt(3.0/14.0)),
                              ((1//1, -1//1, -1//1, -1//1), -sqrt(3.0/14.0)),
                              ((-1//1, 0//1, -1//1, 0//1), 1.0 / sqrt(42.0)),
                              ((-1//1, -1//1, 0//1, 0//1), 1.0 / sqrt(42.0)),
                              ((0//1, -1//1, 0//1, -1//1), 1.0 / sqrt(42.0)),
                              ((-1//1, 0//1, 0//1, -1//1), 1.0 / sqrt(42.0)),
                              ((0//1, 0//1, -1//1, -1//1), 1.0 / sqrt(42.0)),
                              ((0//1, -1//1, -1//1, 0//1), 1.0 / sqrt(42.0))])
            ])
        end

        if J_r == 1//1 && irrep == "[2,1,1]"
            return _build_cg_coeffs(Val(4), j_r, J_r, irrep, multiplicity, dim_irrep, [
                (1//1, 1, [
                              ((1//1, 1//1, -1//1, 0//1), -0.5),
                              ((1//1, 1//1, 0//1, -1//1), 0.5),
                              ((0//1, 1//1, 1//1, -1//1), -1.0 / sqrt(16.0)),
                              ((1//1, 0//1, -1//1, 1//1), 1.0 / sqrt(16.0)),
                              ((1//1, -1//1, 1//1, 0//1), 1.0 / sqrt(16.0)),
                              ((-1//1, 1//1, 0//1, 1//1), -1.0 / sqrt(16.0)),
                              ((1//1, -1//1, 0//1, 1//1), -1.0 / sqrt(16.0)),
                              ((0//1, 1//1, -1//1, 1//1), 1.0 / sqrt(16.0)),
                              ((1//1, 0//1, 1//1, -1//1), -1.0 / sqrt(16.0)),
                              ((-1//1, 1//1, 1//1, 0//1), 1.0 / sqrt(16.0))]),
                (0//1, 1, [
                              ((0//1, 0//1, 1//1, -1//1), 0.5),
                              ((0//1, 0//1, -1//1, 1//1), -0.5),
                              ((0//1, -1//1, 0//1, 1//1), 1.0 / sqrt(16.0)),
                              ((0//1, 1//1, 0//1, -1//1), -1.0 / sqrt(16.0)),
                              ((-1//1, 0//1, 0//1, 1//1), 1.0 / sqrt(16.0)),
                              ((1//1, 0//1, 0//1, -1//1), -1.0 / sqrt(16.0)),
                              ((-1//1, 0//1, 1//1, 0//1), -1.0 / sqrt(16.0)),
                              ((0//1, -1//1, 1//1, 0//1), -1.0 / sqrt(16.0)),
                              ((0//1, 1//1, -1//1, 0//1), 1.0 / sqrt(16.0)),
                              ((1//1, 0//1, -1//1, 0//1), 1.0 / sqrt(16.0))]),
                (-1//1, 1, [
                              ((-1//1, -1//1, 0//1, 1//1), -0.5),
                              ((-1//1, -1//1, 1//1, 0//1), 0.5),
                              ((-1//1, 0//1, 1//1, -1//1), -1.0 / sqrt(16.0)),
                              ((0//1, -1//1, -1//1, 1//1), 1.0 / sqrt(16.0)),
                              ((1//1, -1//1, 0//1, -1//1), 1.0 / sqrt(16.0)),
                              ((-1//1, 1//1, -1//1, 0//1), -1.0 / sqrt(16.0)),
                              ((-1//1, 0//1, -1//1, 1//1), 1.0 / sqrt(16.0)),
                              ((0//1, -1//1, 1//1, -1//1), -1.0 / sqrt(16.0)),
                              ((1//1, -1//1, -1//1, 0//1), -1.0 / sqrt(16.0)),
                              ((-1//1, 1//1, 0//1, -1//1), 1.0 / sqrt(16.0))]),
                (1//1, 2, [
                              ((1//1, -1//1, 1//1, 0//1), -sqrt(3.0/16.0)),
                              ((1//1, 0//1, 1//1, -1//1), sqrt(3.0/16.0)),
                              ((0//1, 1//1, 1//1, -1//1), -sqrt(3.0/16.0)),
                              ((-1//1, 1//1, 1//1, 0//1), sqrt(3.0/16.0)),
                              ((-1//1, 0//1, 1//1, 1//1), -1.0 / sqrt(12.0)),
                              ((0//1, -1//1, 1//1, 1//1), 1.0 / sqrt(12.0)),
                              ((1//1, 0//1, -1//1, 1//1), -1.0 / sqrt(48.0)),
                              ((1//1, -1//1, 0//1, 1//1), 1.0 / sqrt(48.0)),
                              ((-1//1, 1//1, 0//1, 1//1), -1.0 / sqrt(48.0)),
                              ((0//1, 1//1, -1//1, 1//1), 1.0 / sqrt(48.0))]),
                (0//1, 2, [
                              ((1//1, 0//1, 0//1, -1//1), -sqrt(3.0/16.0)),
                              ((0//1, -1//1, 0//1, 1//1), -sqrt(3.0/16.0)),
                              ((-1//1, 0//1, 0//1, 1//1), sqrt(3.0/16.0)),
                              ((0//1, 1//1, 0//1, -1//1), sqrt(3.0/16.0)),
                              ((1//1, -1//1, 0//1, 0//1), 1.0 / sqrt(12.0)),
                              ((-1//1, 1//1, 0//1, 0//1), -1.0 / sqrt(12.0)),
                              ((0//1, -1//1, 1//1, 0//1), 1.0 / sqrt(48.0)),
                              ((-1//1, 0//1, 1//1, 0//1), -1.0 / sqrt(48.0)),
                              ((0//1, 1//1, -1//1, 0//1), -1.0 / sqrt(48.0)),
                              ((1//1, 0//1, -1//1, 0//1), 1.0 / sqrt(48.0))]),
                (-1//1, 2, [
                              ((0//1, -1//1, -1//1, 1//1), -sqrt(3.0/16.0)),
                              ((-1//1, 0//1, -1//1, 1//1), sqrt(3.0/16.0)),
                              ((1//1, -1//1, -1//1, 0//1), sqrt(3.0/16.0)),
                              ((-1//1, 1//1, -1//1, 0//1), -sqrt(3.0/16.0)),
                              ((1//1, 0//1, -1//1, -1//1), -1.0 / sqrt(12.0)),
                              ((0//1, 1//1, -1//1, -1//1), 1.0 / sqrt(12.0)),
                              ((1//1, -1//1, 0//1, -1//1), -1.0 / sqrt(48.0)),
                              ((-1//1, 0//1, 1//1, -1//1), -1.0 / sqrt(48.0)),
                              ((0//1, -1//1, 1//1, -1//1), 1.0 / sqrt(48.0)),
                              ((-1//1, 1//1, 0//1, -1//1), 1.0 / sqrt(48.0))]),
                (1//1, 3, [
                              ((1//1, -1//1, 0//1, 1//1), -1.0 / sqrt(6.0)),
                              ((-1//1, 0//1, 1//1, 1//1), -1.0 / sqrt(6.0)),
                              ((-1//1, 1//1, 0//1, 1//1), 1.0 / sqrt(6.0)),
                              ((0//1, 1//1, -1//1, 1//1), -1.0 / sqrt(6.0)),
                              ((1//1, 0//1, -1//1, 1//1), 1.0 / sqrt(6.0)),
                              ((0//1, -1//1, 1//1, 1//1), 1.0 / sqrt(6.0))]),
                (0//1, 3, [
                              ((0//1, 1//1, -1//1, 0//1), -1.0 / sqrt(6.0)),
                              ((1//1, -1//1, 0//1, 0//1), -1.0 / sqrt(6.0)),
                              ((-1//1, 1//1, 0//1, 0//1), 1.0 / sqrt(6.0)),
                              ((0//1, -1//1, 1//1, 0//1), 1.0 / sqrt(6.0)),
                              ((-1//1, 0//1, 1//1, 0//1), -1.0 / sqrt(6.0)),
                              ((1//1, 0//1, -1//1, 0//1), 1.0 / sqrt(6.0))]),
                (-1//1, 3, [
                              ((1//1, 0//1, -1//1, -1//1), -1.0 / sqrt(6.0)),
                              ((1//1, -1//1, 0//1, -1//1), 1.0 / sqrt(6.0)),
                              ((-1//1, 0//1, 1//1, -1//1), 1.0 / sqrt(6.0)),
                              ((0//1, -1//1, 1//1, -1//1), -1.0 / sqrt(6.0)),
                              ((-1//1, 1//1, 0//1, -1//1), -1.0 / sqrt(6.0)),
                              ((0//1, 1//1, -1//1, -1//1), 1.0 / sqrt(6.0))])
            ])
        end

        if J_r == 1//1 && irrep == "[3,1]"
            return _build_cg_coeffs(Val(4), j_r, J_r, irrep, multiplicity, dim_irrep, [
                (1//1, 1, [
                              ((0//1, 0//1, 0//1, 1//1), -sqrt(3.0/10.0)),
                              ((-1//1, 1//1, 1//1, 0//1), -sqrt(2.0/15.0)),
                              ((1//1, -1//1, 1//1, 0//1), -sqrt(2.0/15.0)),
                              ((1//1, 1//1, -1//1, 0//1), -sqrt(2.0/15.0)),
                              ((1//1, 0//1, -1//1, 1//1), 1.0 / sqrt(30.0)),
                              ((0//1, 1//1, 0//1, 0//1), 1.0 / sqrt(30.0)),
                              ((0//1, -1//1, 1//1, 1//1), 1.0 / sqrt(30.0)),
                              ((1//1, -1//1, 0//1, 1//1), 1.0 / sqrt(30.0)),
                              ((0//1, 1//1, -1//1, 1//1), 1.0 / sqrt(30.0)),
                              ((-1//1, 0//1, 1//1, 1//1), 1.0 / sqrt(30.0)),
                              ((1//1, 0//1, 0//1, 0//1), 1.0 / sqrt(30.0)),
                              ((0//1, 0//1, 1//1, 0//1), 1.0 / sqrt(30.0)),
                              ((-1//1, 1//1, 0//1, 1//1), 1.0 / sqrt(30.0))]),
                (0//1, 1, [
                              ((-1//1, -1//1, 1//1, 1//1), -sqrt(2.0/15.0)),
                              ((-1//1, 1//1, -1//1, 1//1), -sqrt(2.0/15.0)),
                              ((1//1, -1//1, -1//1, 1//1), -sqrt(2.0/15.0)),
                              ((1//1, -1//1, 1//1, -1//1), sqrt(2.0/15.0)),
                              ((-1//1, 1//1, 1//1, -1//1), sqrt(2.0/15.0)),
                              ((1//1, 1//1, -1//1, -1//1), sqrt(2.0/15.0)),
                              ((0//1, 0//1, 1//1, -1//1), -1.0 / sqrt(30.0)),
                              ((1//1, 0//1, 0//1, -1//1), -1.0 / sqrt(30.0)),
                              ((-1//1, 0//1, 0//1, 1//1), 1.0 / sqrt(30.0)),
                              ((0//1, 1//1, 0//1, -1//1), -1.0 / sqrt(30.0)),
                              ((0//1, -1//1, 0//1, 1//1), 1.0 / sqrt(30.0)),
                              ((0//1, 0//1, -1//1, 1//1), 1.0 / sqrt(30.0))]),
                (-1//1, 1, [
                              ((0//1, 0//1, 0//1, -1//1), sqrt(3.0/10.0)),
                              ((1//1, -1//1, -1//1, 0//1), sqrt(2.0/15.0)),
                              ((-1//1, -1//1, 1//1, 0//1), sqrt(2.0/15.0)),
                              ((-1//1, 1//1, -1//1, 0//1), sqrt(2.0/15.0)),
                              ((0//1, -1//1, 0//1, 0//1), -1.0 / sqrt(30.0)),
                              ((0//1, 1//1, -1//1, -1//1), -1.0 / sqrt(30.0)),
                              ((-1//1, 0//1, 1//1, -1//1), -1.0 / sqrt(30.0)),
                              ((0//1, -1//1, 1//1, -1//1), -1.0 / sqrt(30.0)),
                              ((-1//1, 0//1, 0//1, 0//1), -1.0 / sqrt(30.0)),
                              ((1//1, 0//1, -1//1, -1//1), -1.0 / sqrt(30.0)),
                              ((-1//1, 1//1, 0//1, -1//1), -1.0 / sqrt(30.0)),
                              ((1//1, -1//1, 0//1, -1//1), -1.0 / sqrt(30.0)),
                              ((0//1, 0//1, -1//1, 0//1), -1.0 / sqrt(30.0))]),
                (1//1, 2, [
                              ((0//1, 0//1, 1//1, 0//1), sqrt(4.0/15.0)),
                              ((1//1, 1//1, 0//1, -1//1), sqrt(3.0/20.0)),
                              ((1//1, -1//1, 0//1, 1//1), sqrt(5.0/48.0)),
                              ((-1//1, 1//1, 0//1, 1//1), sqrt(5.0/48.0)),
                              ((-1//1, 0//1, 1//1, 1//1), -1.0 / sqrt(15.0)),
                              ((1//1, 0//1, 0//1, 0//1), -1.0 / sqrt(15.0)),
                              ((0//1, 1//1, 0//1, 0//1), -1.0 / sqrt(15.0)),
                              ((0//1, -1//1, 1//1, 1//1), -1.0 / sqrt(15.0)),
                              ((0//1, 1//1, 1//1, -1//1), -sqrt(3.0/80.0)),
                              ((1//1, 0//1, 1//1, -1//1), -sqrt(3.0/80.0)),
                              ((1//1, 1//1, -1//1, 0//1), sqrt(3.0/180.0)),
                              ((0//1, 1//1, -1//1, 1//1), -0.0645497224367903),
                              ((1//1, -1//1, 1//1, 0//1), -0.0645497224367903),
                              ((-1//1, 1//1, 1//1, 0//1), -0.0645497224367903),
                              ((1//1, 0//1, -1//1, 1//1), -0.0645497224367903)]),
                (0//1, 2, [
                              ((1//1, 1//1, -1//1, -1//1), -sqrt(4.0/15.0)),
                              ((-1//1, -1//1, 1//1, 1//1), sqrt(4.0/15.0)),
                              ((1//1, -1//1, 1//1, -1//1), 1.0 / sqrt(15.0)),
                              ((1//1, -1//1, -1//1, 1//1), -1.0 / sqrt(15.0)),
                              ((-1//1, 1//1, -1//1, 1//1), -1.0 / sqrt(15.0)),
                              ((-1//1, 1//1, 1//1, -1//1), 1.0 / sqrt(15.0)),
                              ((0//1, -1//1, 1//1, 0//1), -sqrt(3.0/80.0)),
                              ((0//1, 1//1, -1//1, 0//1), sqrt(3.0/80.0)),
                              ((1//1, 0//1, -1//1, 0//1), sqrt(3.0/80.0)),
                              ((-1//1, 0//1, 1//1, 0//1), -sqrt(3.0/80.0)),
                              ((0//1, 0//1, 1//1, -1//1), -sqrt(3.0/180.0)),
                              ((0//1, 0//1, -1//1, 1//1), sqrt(3.0/180.0)),
                              ((1//1, 0//1, 0//1, -1//1), 0.0645497224367903),
                              ((-1//1, 0//1, 0//1, 1//1), -0.0645497224367903),
                              ((0//1, 1//1, 0//1, -1//1), 0.0645497224367903),
                              ((0//1, -1//1, 0//1, 1//1), -0.0645497224367903)]),
                (-1//1, 2, [
                              ((0//1, 0//1, -1//1, 0//1), -sqrt(4.0/15.0)),
                              ((-1//1, -1//1, 0//1, 1//1), -sqrt(3.0/20.0)),
                              ((1//1, -1//1, 0//1, -1//1), -sqrt(5.0/48.0)),
                              ((-1//1, 1//1, 0//1, -1//1), -sqrt(5.0/48.0)),
                              ((1//1, 0//1, -1//1, -1//1), 1.0 / sqrt(15.0)),
                              ((0//1, -1//1, 0//1, 0//1), 1.0 / sqrt(15.0)),
                              ((-1//1, 0//1, 0//1, 0//1), 1.0 / sqrt(15.0)),
                              ((0//1, 1//1, -1//1, -1//1), 1.0 / sqrt(15.0)),
                              ((-1//1, 0//1, -1//1, 1//1), sqrt(3.0/80.0)),
                              ((0//1, -1//1, -1//1, 1//1), sqrt(3.0/80.0)),
                              ((-1//1, -1//1, 1//1, 0//1), -sqrt(3.0/180.0)),
                              ((0//1, -1//1, 1//1, -1//1), 0.0645497224367903),
                              ((1//1, -1//1, -1//1, 0//1), 0.0645497224367903),
                              ((-1//1, 1//1, -1//1, 0//1), 0.0645497224367903),
                              ((-1//1, 0//1, 1//1, -1//1), 0.0645497224367903)]),
                (1//1, 3, [
                              ((0//1, 1//1, 0//1, 0//1), -1.0 / sqrt(5.0)),
                              ((1//1, 0//1, 0//1, 0//1), 1.0 / sqrt(5.0)),
                              ((0//1, 1//1, 1//1, -1//1), sqrt(9.0/80.0)),
                              ((1//1, 0//1, -1//1, 1//1), -sqrt(9.0/80.0)),
                              ((1//1, 0//1, 1//1, -1//1), -sqrt(9.0/80.0)),
                              ((0//1, 1//1, -1//1, 1//1), sqrt(9.0/80.0)),
                              ((-1//1, 0//1, 1//1, 1//1), -1.0 / sqrt(20.0)),
                              ((0//1, -1//1, 1//1, 1//1), 1.0 / sqrt(20.0)),
                              ((-1//1, 1//1, 0//1, 1//1), 1.0 / sqrt(80.0)),
                              ((1//1, -1//1, 0//1, 1//1), -1.0 / sqrt(80.0)),
                              ((1//1, -1//1, 1//1, 0//1), -1.0 / sqrt(80.0)),
                              ((-1//1, 1//1, 1//1, 0//1), 1.0 / sqrt(80.0))]),
                (0//1, 3, [
                              ((1//1, -1//1, -1//1, 1//1), 1.0 / sqrt(5.0)),
                              ((-1//1, 1//1, 1//1, -1//1), -1.0 / sqrt(5.0)),
                              ((-1//1, 1//1, -1//1, 1//1), -1.0 / sqrt(5.0)),
                              ((1//1, -1//1, 1//1, -1//1), 1.0 / sqrt(5.0)),
                              ((1//1, -1//1, 0//1, 0//1), -1.0 / sqrt(20.0)),
                              ((-1//1, 1//1, 0//1, 0//1), 1.0 / sqrt(20.0)),
                              ((1//1, 0//1, 0//1, -1//1), -1.0 / sqrt(80.0)),
                              ((0//1, 1//1, 0//1, -1//1), 1.0 / sqrt(80.0)),
                              ((-1//1, 0//1, 1//1, 0//1), 1.0 / sqrt(80.0)),
                              ((-1//1, 0//1, 0//1, 1//1), 1.0 / sqrt(80.0)),
                              ((0//1, -1//1, 1//1, 0//1), -1.0 / sqrt(80.0)),
                              ((0//1, 1//1, -1//1, 0//1), 1.0 / sqrt(80.0)),
                              ((0//1, -1//1, 0//1, 1//1), -1.0 / sqrt(80.0)),
                              ((1//1, 0//1, -1//1, 0//1), -1.0 / sqrt(80.0))]),
                (-1//1, 3, [
                              ((0//1, -1//1, 0//1, 0//1), 1.0 / sqrt(5.0)),
                              ((-1//1, 0//1, 0//1, 0//1), -1.0 / sqrt(5.0)),
                              ((0//1, -1//1, -1//1, 1//1), -sqrt(9.0/80.0)),
                              ((-1//1, 0//1, 1//1, -1//1), sqrt(9.0/80.0)),
                              ((-1//1, 0//1, -1//1, 1//1), sqrt(9.0/80.0)),
                              ((0//1, -1//1, 1//1, -1//1), -sqrt(9.0/80.0)),
                              ((0//1, 1//1, -1//1, -1//1), -1.0 / sqrt(20.0)),
                              ((1//1, 0//1, -1//1, -1//1), 1.0 / sqrt(20.0)),
                              ((-1//1, 1//1, -1//1, 0//1), 1.0 / sqrt(80.0)),
                              ((1//1, -1//1, 0//1, -1//1), -1.0 / sqrt(80.0)),
                              ((-1//1, 1//1, 0//1, -1//1), 1.0 / sqrt(80.0)),
                              ((1//1, -1//1, -1//1, 0//1), -1.0 / sqrt(80.0))])
            ])
        end

        if J_r == 0//1 && irrep == "[2,2]"
            return _build_cg_coeffs(Val(4), j_r, J_r, irrep, multiplicity, dim_irrep, [
                (0//1, 1, [
                              ((1//1, 1//1, -1//1, -1//1), 1.0 / sqrt(9.0)),
                              ((-1//1, -1//1, 1//1, 1//1), 1.0 / sqrt(9.0)),
                              ((0//1, 0//1, -1//1, 1//1), 1.0 / sqrt(9.0)),
                              ((-1//1, 1//1, 0//1, 0//1), 1.0 / sqrt(9.0)),
                              ((0//1, 0//1, 1//1, -1//1), 1.0 / sqrt(9.0)),
                              ((1//1, -1//1, 0//1, 0//1), 1.0 / sqrt(9.0)),
                              ((1//1, 0//1, 0//1, -1//1), -1.0 / sqrt(36.0)),
                              ((1//1, 0//1, -1//1, 0//1), -1.0 / sqrt(36.0)),
                              ((0//1, 1//1, 0//1, -1//1), -1.0 / sqrt(36.0)),
                              ((0//1, 1//1, -1//1, 0//1), -1.0 / sqrt(36.0)),
                              ((-1//1, 0//1, 0//1, 1//1), -1.0 / sqrt(36.0)),
                              ((0//1, -1//1, 0//1, 1//1), -1.0 / sqrt(36.0)),
                              ((0//1, -1//1, 1//1, 0//1), -1.0 / sqrt(36.0)),
                              ((-1//1, 0//1, 1//1, 0//1), -1.0 / sqrt(36.0)),
                              ((-1//1, 1//1, -1//1, 1//1), -1.0 / sqrt(36.0)),
                              ((-1//1, 1//1, 1//1, -1//1), -1.0 / sqrt(36.0)),
                              ((1//1, -1//1, -1//1, 1//1), -1.0 / sqrt(36.0)),
                              ((1//1, -1//1, 1//1, -1//1), -1.0 / sqrt(36.0))]),
                (0//1, 2, [
                              ((-1//1, 1//1, 1//1, -1//1), 1.0 / sqrt(12.0)),
                              ((1//1, -1//1, -1//1, 1//1), 1.0 / sqrt(12.0)),
                              ((-1//1, 0//1, 1//1, 0//1), -1.0 / sqrt(12.0)),
                              ((0//1, 1//1, 0//1, -1//1), -1.0 / sqrt(12.0)),
                              ((0//1, -1//1, 0//1, 1//1), -1.0 / sqrt(12.0)),
                              ((1//1, 0//1, -1//1, 0//1), -1.0 / sqrt(12.0)),
                              ((0//1, 1//1, -1//1, 0//1), 1.0 / sqrt(12.0)),
                              ((-1//1, 0//1, 0//1, 1//1), 1.0 / sqrt(12.0)),
                              ((1//1, 0//1, 0//1, -1//1), 1.0 / sqrt(12.0)),
                              ((0//1, -1//1, 1//1, 0//1), 1.0 / sqrt(12.0)),
                              ((-1//1, 1//1, -1//1, 1//1), -1.0 / sqrt(12.0)),
                              ((1//1, -1//1, 1//1, -1//1), -1.0 / sqrt(12.0))])
            ])
        end

        if J_r == 0//1 && irrep == "[4]"
            return _build_cg_coeffs(Val(4), j_r, J_r, irrep, multiplicity, dim_irrep, [
                (0//1, 1, [
                              ((0//1, 0//1, 0//1, 0//1), 1.0 / sqrt(5.0)),
                              ((-1//1, 1//1, 1//1, -1//1), sqrt(4.0/45.0)),
                              ((1//1, -1//1, -1//1, 1//1), sqrt(4.0/45.0)),
                              ((1//1, 1//1, -1//1, -1//1), sqrt(4.0/45.0)),
                              ((-1//1, -1//1, 1//1, 1//1), sqrt(4.0/45.0)),
                              ((1//1, -1//1, 1//1, -1//1), sqrt(4.0/45.0)),
                              ((-1//1, 1//1, -1//1, 1//1), sqrt(4.0/45.0)),
                              ((0//1, 1//1, 0//1, -1//1), -1.0 / sqrt(45.0)),
                              ((1//1, 0//1, -1//1, 0//1), -1.0 / sqrt(45.0)),
                              ((0//1, -1//1, 0//1, 1//1), -1.0 / sqrt(45.0)),
                              ((-1//1, 0//1, 1//1, 0//1), -1.0 / sqrt(45.0)),
                              ((0//1, 0//1, 1//1, -1//1), -1.0 / sqrt(45.0)),
                              ((1//1, -1//1, 0//1, 0//1), -1.0 / sqrt(45.0)),
                              ((-1//1, 1//1, 0//1, 0//1), -1.0 / sqrt(45.0)),
                              ((0//1, 0//1, -1//1, 1//1), -1.0 / sqrt(45.0)),
                              ((-1//1, 0//1, 0//1, 1//1), -1.0 / sqrt(45.0)),
                              ((0//1, 1//1, -1//1, 0//1), -1.0 / sqrt(45.0)),
                              ((1//1, 0//1, 0//1, -1//1), -1.0 / sqrt(45.0)),
                              ((0//1, -1//1, 1//1, 0//1), -1.0 / sqrt(45.0))])
            ])
        end

    else
        throw(ArgumentError("CG coefficients not yet implemented for N=$N, j=$j_r"))
    end
end

# ============ Multi-species isospin decomposition ============

"""
    MultiIsospinEntry

One multi-species isospin-decomposition entry. Schur–Weyl duality determines (J_s, κ_s) within every species; the J_s are coupled by SU(2) CG coefficients to total isospin I. The full permutation group is S_{N₁}×⋯×S_{Nₖ}, whose irrep is the tensor product of factor irreps.

Fields:
- J_tuple::NTuple{K,Rational{Int}}: coupled total isospin of each species
- κ_tuple::NTuple{K,String}: S_N irrep for each species
- coupling_mult::Int: multiplicity (number of paths) for coupling J_s to I through CG coefficients
- internal_mult::Int: product of intrinsic multiplicities of all species ∏ₛ mult_s
- coupling_paths::Vector{Tuple}: intermediate-isospin paths in a left-associated coupling scheme
- internal_multiplicities::Tuple: intrinsic multiplicities of fixed (J_s, κ_s) in each species
"""
struct MultiIsospinEntry
    J_tuple::NTuple{N,Rational{Int}} where N
    κ_tuple::NTuple{N,String} where N
    coupling_mult::Int
    internal_mult::Int
    coupling_paths::Vector{Tuple}
    internal_multiplicities::Tuple
end

"""
    MultiIsospinDecomposition

Multi-species isospin-decomposition result.

Fields:
- species::Vector{Int}: particle counts of all species [N₁, N₂, ...]
- isospins::Vector{Rational{Int}}: single-particle isospins of all species [j₁, j₂, ...]
- I::Rational{Int}: total isospin
- entries::Vector{MultiIsospinEntry}: all valid decomposition entries
"""
struct MultiIsospinDecomposition
    species::Vector{Int}
    isospins::Vector{Rational{Int}}
    I::Rational{Int}
    entries::Vector{MultiIsospinEntry}
end

function Base.show(io::IO, entry::MultiIsospinEntry)
    J_str = join(string.(entry.J_tuple), ", ")
    κ_str = join(entry.κ_tuple, ", ")
    total_mult = entry.coupling_mult * entry.internal_mult
    print(io, "(Js=($J_str), κs=($κ_str), cg=$(entry.coupling_mult), int=$(entry.internal_mult), tot=$total_mult)")
end

function Base.show(io::IO, d::MultiIsospinDecomposition)
    k = length(d.species)
    println(io, "MultiIsospinDecomposition (species=$(d.species), isospins=$(d.isospins), I=$(d.I)):")
    if isempty(d.entries)
        println(io, "  (no entries)")
    else
        println(io, "  " * "-"^72)
        for entry in d.entries
            println(io, "  $entry")
        end
    end
end

# ---- Multi-species single-species decomposition helper ----

function _single_species_entries(N::Int, j::Rational{Int})
    if N == 1
        # Single-particle species: trivial decomposition.
        return [(j, "[1]", 1)]
    end
    decomp = isospin_decomposition(N, j)
    return [(J, κ, mult) for (J, κ, mult) in decomp.entries]
end

# ---- SU(2) coupling multiplicities ----

"""
    _coupling_paths(Js::Vector{Rational{Int}}, I::Rational{Int}) -> Vector{Tuple}

List all independent paths coupling J₁⊗J₂⊗⋯⊗Jₖ to total I.
Use a left-associated scheme (⋯((J₁⊗J₂)⊗J₃)⊗⋯); the tuple stores intermediate isospins J₁₂, J₁₂₃, ... before final I.
"""
function _coupling_paths(Js::Vector{Rational{Int}}, I::Rational{Int})
    isempty(Js) && return I == 0//1 ? Tuple[()] : Tuple[]
    paths = Tuple[]
    _collect_coupling_paths!(paths, Js, 2, Js[1], I, Rational{Int}[])
    return paths
end

function _collect_coupling_paths!(paths::Vector{Tuple}, Js::Vector{Rational{Int}},
                                  idx::Int, current::Rational{Int},
                                  target::Rational{Int}, intermediates::Vector{Rational{Int}})
    if idx > length(Js)
        current == target && push!(paths, Tuple(intermediates))
        return nothing
    end
    lo = abs(current - Js[idx])
    hi = current + Js[idx]
    for J_interm in lo:hi
        keep = idx < length(Js)
        keep && push!(intermediates, J_interm)
        _collect_coupling_paths!(paths, Js, idx + 1, J_interm, target, intermediates)
        keep && pop!(intermediates)
    end
    return nothing
end

_coupling_multiplicity(Js::Vector{Rational{Int}}, I::Rational{Int}) =
    length(_coupling_paths(Js, I))

"""
    multi_isospin_decomposition(species::Vector{Int},
                                 isospins::Vector{<:Union{Rational{Int},Int}},
                                 I::Union{Rational{Int},Int})
        -> MultiIsospinDecomposition

Return the isospin-decomposition result for a multi-species system.

# Arguments
- `species`: particle counts of all species [N₁, N₂, ..., Nₖ]
- `isospins`: single-particle isospins of all species [j₁, j₂, ..., jₖ]
- `I`: total isospin

# Algorithm
1. Perform Schur–Weyl decomposition for each species (use the trivial decomposition directly for N=1).
2. Take the Cartesian product of species entries.
3. For each (J₁,...,Jₖ), determine whether SU(2) CG coupling can yield total I.
4. Coupling multiplicity = number of independent coupling paths (iterated CG).

# Example
```julia
# Single species (backward compatible).
d = multi_isospin_decomposition([2], [1//2], 1//1)

# ππN: two π (I=1) plus one nucleon (I=1/2) → total I=1/2
d = multi_isospin_decomposition([2, 1], [1//1, 1//2], 1//2)
```
"""
function multi_isospin_decomposition(species::Vector{Int},
                                      isospins::Vector{<:Union{Rational{Int},Int}},
                                      I::Union{Rational{Int},Int})
    I_r = Rational{Int}(I)
    k = length(species)
    length(isospins) == k ||
        throw(ArgumentError("isospins length ($(length(isospins))) must equal species length ($k)"))

    isospin_r = Rational{Int}.(isospins)

    # Decompose each species separately.
    per_species = [_single_species_entries(species[s], isospin_r[s]) for s in 1:k]

    # Cartesian product + coupling test.
    entries = MultiIsospinEntry[]
    Js = Rational{Int}[]
    κs = String[]
    mults = Int[]
    _build_multi_entries!(entries, per_species, I_r, Js, κs, mults)
    return MultiIsospinDecomposition(species, isospin_r, I_r, entries)
end

function _build_multi_entries!(entries::Vector{MultiIsospinEntry},
                                per_species::Vector,
                                I::Rational{Int},
                                Js::Vector, κs::Vector,
                                mults::Vector{Int})
    s_depth = length(Js) + 1
    if s_depth <= length(per_species)
        for (J, κ, mult) in per_species[s_depth]
            push!(Js, J)
            push!(κs, κ)
            push!(mults, mult)
            _build_multi_entries!(entries, per_species, I, Js, κs, mults)
            pop!(mults)
            pop!(κs)
            pop!(Js)
        end
    else
        paths = _coupling_paths(Rational{Int}.(Js), I)
        if !isempty(paths)
            internal_mult = prod(mults; init=1)
            push!(entries, MultiIsospinEntry(Tuple(Js), Tuple(κs), length(paths),
                                             internal_mult, paths, Tuple(mults)))
        end
    end
end

# ============================================================
# Multi-species direct-product group utilities (S_{N₁} × ⋯ × S_{Nₖ})
# ============================================================

"""
    _build_full_permutation(species::Vector{Int}, per_s_indices::NTuple{K,Int}) where K -> Vector{Int}

Given a direct-product group element specified by each species S_{Nᵢ} permutation index, return the full N-particle permutation vector.
The permutation convention matches SN_ELEMENTS: p[i] is the destination of original position i.
"""
function _build_full_permutation(species::Vector{Int}, per_s_indices::NTuple{K,Int}) where K
    N = sum(species)
    result = Vector{Int}(undef, N)
    offset = 0
    for k in 1:K
        Nk = species[k]
        sk = SN_ELEMENTS[Nk][per_s_indices[k]]
        for i in 1:Nk
            result[offset + i] = offset + sk[i]
        end
        offset += Nk
    end
    return result
end

"""
    _decompose_multi_species_permutation(p::Vector{Int}, species::Vector{Int}) -> NTuple{K,Int}

_build_full_permutation Inverse operation. Given full permutation p (permuting only within species),
return tuple per_s_idx of indices in SN_ELEMENTS for all species.
"""
function _decompose_multi_species_permutation(p::Vector{Int}, species::Vector{Int})
    K = length(species)
    idxs = Vector{Int}(undef, K)
    offset = 0
    for k in 1:K
        Nk = species[k]
        sk = [p[offset + i] - offset for i in 1:Nk]
        idxs[k] = get_SN_element_index(Nk, sk)
        offset += Nk
    end
    return NTuple{K,Int}(idxs)
end

"""
    _product_group_generators(species::Vector{Int}) -> Vector

Generate all elements of the direct-product group G = S_{N₁}×⋯×S_{Nₖ}.

Return a Vector of NamedTuples, each containing:
- `s_full::Vector{Int}`: N full N-particle permutation
- `per_s::NTuple{K,Vector{Int}}`: permutation of each species (1-indexed within species)
- `per_s_idx::NTuple{K,Int}`: SN_ELEMENTS index of each species
- `flat_idx::Int`: flattened index (1..total_dim)
"""
function _product_group_generators(species::Vector{Int})
    K = length(species)
    per_sizes = [length(SN_ELEMENTS[species[k]]) for k in 1:K]
    total_dim = prod(per_sizes)

    result = Vector{NamedTuple{(:s_full, :per_s, :per_s_idx, :flat_idx),
                                Tuple{Vector{Int}, NTuple{K,Vector{Int}}, NTuple{K,Int}, Int}}}(undef, total_dim)

    # Generate all combinations using Cartesian indices.
    counters = ones(Int, K)
    for flat in 1:total_dim
        per_s = ntuple(k -> SN_ELEMENTS[species[k]][counters[k]], K)
        s_full = _build_full_permutation(species, ntuple(k -> counters[k], K))

        result[flat] = (s_full=s_full, per_s=per_s,
                         per_s_idx=ntuple(k -> counters[k], K), flat_idx=flat)

        # Increment counter (column-major: last index fastest or first index fastest?)
        # Use simple carry logic; the first species varies fastest (mixed radix).
        for k in 1:K
            counters[k] += 1
            if counters[k] <= per_sizes[k]
                break
            else
                counters[k] = 1
            end
        end
    end

    return result
end

"""
    _kappa_tuple_dim(species::Vector{Int}, κ_tuple) -> Int

Return the dimension of a direct-product irrep: ∏_k dim([κ_k])。
"""
function _kappa_tuple_dim(species::Vector{Int}, κ_tuple)
    dim = 1
    for k in 1:length(species)
        dim *= get_SN_irrep_dim(species[k], κ_tuple[k])
    end
    return dim
end

@inline function _next_carrier_component(remainder::Int, dim::Int)
    return mod(remainder, dim) + 1, remainder ÷ dim
end

"""
    _carrier_index_tuple(species, κ_tuple, flat_index) -> Tuple

Convert a linear direct-product carrier-space index to per-species indices `(a₁,a₂,...)`。
Species-1 index varies fastest; this convention is used for both projection matrices and user interaction matrices.
"""
function _carrier_index_tuple(species::Vector{Int}, κ_tuple, flat_index::Int)
    total_dim = _kappa_tuple_dim(species, κ_tuple)
    1 <= flat_index <= total_dim || throw(BoundsError(1:total_dim, flat_index))
    indices = Vector{Int}(undef, length(species))
    remainder = flat_index - 1
    for k in eachindex(species)
        dim_k = get_SN_irrep_dim(species[k], κ_tuple[k])
        indices[k], remainder = _next_carrier_component(remainder, dim_k)
    end
    return Tuple(indices)
end

_carrier_index_tuples(species::Vector{Int}, κ_tuple) =
    [_carrier_index_tuple(species, κ_tuple, a)
     for a in 1:_kappa_tuple_dim(species, κ_tuple)]

# ============ Charge-basis expansion of physical isospin subchannels ============

@inline _su2_factorial(x::Rational{Int}) = factorial(Int(x))

"""
    _su2_clebsch_gordan(j1, m1, j2, m2, J, M) -> Float64

SU(2) Clebsch–Gordan coefficient in the Condon–Shortley convention
`⟨j₁m₁,j₂m₂|JM⟩`。Arguments are rational values representing integers or half-integers.
"""
function _su2_clebsch_gordan(j1::Rational{Int}, m1::Rational{Int},
                             j2::Rational{Int}, m2::Rational{Int},
                             J::Rational{Int}, M::Rational{Int})
    m1 + m2 == M || return 0.0
    abs(m1) <= j1 && abs(m2) <= j2 && abs(M) <= J || return 0.0
    abs(j1 - j2) <= J <= j1 + j2 || return 0.0

    prefactor = sqrt(Float64(
        (2J + 1) *
        _su2_factorial(J + j1 - j2) *
        _su2_factorial(J - j1 + j2) *
        _su2_factorial(j1 + j2 - J) /
        _su2_factorial(j1 + j2 + J + 1) *
        _su2_factorial(J + M) * _su2_factorial(J - M) *
        _su2_factorial(j1 + m1) * _su2_factorial(j1 - m1) *
        _su2_factorial(j2 + m2) * _su2_factorial(j2 - m2)
    ))

    k_min = maximum((0, Int(j2 - J - m1), Int(j1 + m2 - J)))
    k_max = minimum((Int(j1 + j2 - J), Int(j1 - m1), Int(j2 + m2)))
    k_min <= k_max || return 0.0
    series = 0.0
    for k in k_min:k_max
        denominator = factorial(k) *
                      _su2_factorial(j1 + j2 - J - k) *
                      _su2_factorial(j1 - m1 - k) *
                      _su2_factorial(j2 + m2 - k) *
                      _su2_factorial(J - j2 + m1 + k) *
                      _su2_factorial(J - j1 - m2 + k)
        series += isodd(k) ? -1 / denominator : 1 / denominator
    end
    return prefactor * series
end

_isospin_m_values(J::Rational{Int}) = collect((-J):1:J)

function _coupling_path_coefficient(Js::Tuple, Ms::Tuple,
                                    path::Tuple, I::Rational{Int},
                                    M::Rational{Int})
    length(Js) == length(Ms) || throw(DimensionMismatch("Js and Ms must have equal length"))
    isempty(Js) && return I == 0 && M == 0 ? 1.0 : 0.0
    length(Js) == 1 && return Js[1] == I && Ms[1] == M ? 1.0 : 0.0

    current_J = Js[1]
    current_M = Ms[1]
    coefficient = 1.0
    for s in 2:length(Js)
        target_J = s < length(Js) ? path[s - 1] : I
        target_M = current_M + Ms[s]
        coefficient *= _su2_clebsch_gordan(
            current_J, current_M, Js[s], Ms[s], target_J, target_M)
        abs(coefficient) < 1e-14 && return 0.0
        current_J = target_J
        current_M = target_M
    end
    return current_M == M ? coefficient : 0.0
end

function _species_charge_coefficients(N::Int, j::Rational{Int}, J::Rational{Int},
                                      κ::String, μ::Int, M::Rational{Int}, a::Int)
    if N == 1
        J == j && κ == "[1]" && μ == 1 && a == 1 ||
            throw(ArgumentError("invalid one-particle isospin state"))
        return Dict{Tuple,ComplexF64}((M,) => 1.0 + 0.0im)
    end
    cg = charge_to_isospin_cg(N, j, J, κ; multiplicity=μ)
    return Dict{Tuple,ComplexF64}(Tuple(config) => coefficient
                                  for (config, coefficient) in cg.coeffs[M][a])
end

"""
    _subchannel_charge_coefficients(species, isospins, I, sub, a; M=I)

Return expansion coefficients of physical state `|I,M;κ,a,r⟩` in ordered charge basis `|m₁,...,m_N⟩`. Species use a left-associated coupling path, particle order follows `species`, and composite carrier index `a` uses the `_carrier_index_tuple` ordering.
"""
function _subchannel_charge_coefficients(species::Vector{Int},
                                         isospins::Vector{Rational{Int}},
                                         I::Rational{Int}, sub, a::Int;
                                         M::Rational{Int}=I)
    K = length(species)
    length(isospins) == K || throw(DimensionMismatch("species and isospins must match"))
    κ_tuple = sub.κ isa Tuple ? sub.κ : (sub.κ,)
    a_tuple = _carrier_index_tuple(species, κ_tuple, a)
    result = Dict{Tuple,ComplexF64}()

    m_ranges = map(_isospin_m_values, sub.J_tuple)
    for Ms_raw in Iterators.product(m_ranges...)
        Ms = Tuple(Ms_raw)
        coupling = _coupling_path_coefficient(sub.J_tuple, Ms,
                                              sub.coupling_path, I, M)
        abs(coupling) < 1e-14 && continue
        partial = Dict{Tuple,ComplexF64}(() => ComplexF64(coupling))
        for s in 1:K
            species_coeffs = _species_charge_coefficients(
                species[s], isospins[s], sub.J_tuple[s], κ_tuple[s],
                sub.multiplicity_tuple[s], Ms[s], a_tuple[s])
            next_partial = Dict{Tuple,ComplexF64}()
            for (left_state, left_coefficient) in partial,
                (right_state, right_coefficient) in species_coeffs
                state = (left_state..., right_state...)
                next_partial[state] = get(next_partial, state, 0.0 + 0.0im) +
                                      left_coefficient * right_coefficient
            end
            partial = next_partial
        end
        for (state, coefficient) in partial
            result[state] = get(result, state, 0.0 + 0.0im) + coefficient
        end
    end
    filter!(pair -> abs(last(pair)) >= 1e-12, result)
    return result
end

"""
    _multi_R_matrix_element(species::Vector{Int}, κ_tuple, per_s_idx::NTuple{K,Int},
                            a_flat::Int, b_flat::Int) where K -> Float64

Return direct-product representation matrix element R^{[κ]}(s)_{b,a} = ∏_k R^{[κ_k]}(s_k)_{b_k, a_k}.
a_flat / b_flat are flattened multi-indices (column-major: species 1 varies fastest).
"""
function _multi_R_matrix_element(species::Vector{Int}, κ_tuple, per_s_idx::NTuple{K,Int},
                                  a_flat::Int, b_flat::Int) where K
    val = 1.0
    a_rem = a_flat - 1
    b_rem = b_flat - 1
    for k in 1:K
        dim_k = get_SN_irrep_dim(species[k], κ_tuple[k])
        a_k, a_rem = _next_carrier_component(a_rem, dim_k)
        b_k, b_rem = _next_carrier_component(b_rem, dim_k)
        R_k = get_SN_irrep_matrix(species[k], κ_tuple[k], per_s_idx[k])
        val *= R_k[b_k, a_k]
    end
    return val
end

# (end of IsospinSpace — this file is included directly into NPHFforFVE.jl)
