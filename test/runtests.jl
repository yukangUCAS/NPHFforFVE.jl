using NPHFforFVE
using Test
using StaticArrays
using LinearAlgebra

const M = NPHFforFVE.Momentum  # SVector{3, Int}

@testset "single_particle_basis" begin
    # Ncut = 0: 只有原点
    b0 = single_particle_basis(0)
    @test length(b0) == 1
    @test b0[1] == M(0, 0, 0)

    # Ncut = 1: 原点 + 6 个 (±1,0,0) 排列
    b1 = single_particle_basis(1)
    @test length(b1) == 7

    # Ncut = 2: 额外 12 个 (±1,±1,0) 排列
    b2 = single_particle_basis(2)
    @test length(b2) == 19

    # 验证排序和截断条件
    for b in [b0, b1, b2]
        for n in b
            @test n[1]^2 + n[2]^2 + n[3]^2 <= (b == b0 ? 0 : b == b1 ? 1 : 2)
        end
        # 验证字典序排序
        @test issorted(b)
    end
end

@testset "预设总动量常量" begin
    @test D000 == M(0, 0, 0)
    @test D001 == M(0, 0, 1)
    @test D011 == M(0, 1, 1)
    @test D111 == M(1, 1, 1)
end

@testset "momentum_states N=1" begin
    d = M(0, 0, 0)
    states = collect(momentum_states(1, Ncut=2, d=d))
    @test length(states) == 1
    @test states[1][1] == d

    states_b = collect(momentum_states(1, Ncut=2, d=d, particle_type=:boson))
    @test length(states_b) == 1

    states_f = collect(momentum_states(1, Ncut=2, d=d, particle_type=:fermion))
    @test length(states_f) == 1

    # 默认 d=(0,0,0)
    states_default = collect(momentum_states(1, Ncut=2))
    @test length(states_default) == 1
end

@testset "momentum_states N=2, d=(0,0,0)" begin
    # d=0, Ncut=2: n₁ + n₂ = 0 => n₂ = -n₁，每个 n₁ 在基中即合法
    d = M(0, 0, 0)
    basis = single_particle_basis(2)
    n_basis = length(basis)  # 19

    # 可区分粒子：所有 n₁, n₂=-n₁ 的组合
    states = collect(momentum_states(2, Ncut=2, d=d))
    @test length(states) == n_basis

    # 玻色子：无序 {n₁, n₂}，n₁ ≤ n₂
    states_b = collect(momentum_states(2, Ncut=2, d=d, particle_type=:boson))
    # 每对 {n, -n} 且 n ≤ -n => n_x ≤ 0，或者 n=0
    for s in states_b
        @test s[1] <= s[2]  # 排序约束
        @test s[1] + s[2] == d  # 总动量约束
    end

    # 费米子：无序 {n₁, n₂}, n₁ < n₂, 不含 n₁=n₂
    states_f = collect(momentum_states(2, Ncut=2, d=d, particle_type=:fermion))
    for s in states_f
        @test s[1] < s[2]  # 严格排序
        @test s[1] + s[2] == d
    end
    # n=(0,0,0) 时 n₂=-n=(0,0,0)，相等，应被排除
    @test !any(s -> s[1] == s[2], states_f)
end

@testset "momentum_states N=2, d≠(0,0,0)" begin
    d = M(1, 0, 0)
    states = collect(momentum_states(2, Ncut=2, d=d))
    @test length(states) > 0
    for s in states
        @test s[1] + s[2] == d
        @test sum(abs2, s[1]) <= 2
        @test sum(abs2, s[2]) <= 2
    end
end

@testset "momentum_states N=3, d=(0,0,0), Ncut=2" begin
    d = M(0, 0, 0)
    states = collect(momentum_states(3, Ncut=2, d=d))
    for s in states
        @test s[1] + s[2] + s[3] == d
        for n in s
            @test sum(abs2, n) <= 2
        end
    end
    # 验证计数一致性
    @test count_momentum_states(3, Ncut=2, d=d) == length(states)

    states_b = collect(momentum_states(3, Ncut=2, d=d, particle_type=:boson))
    for s in states_b
        @test s[1] <= s[2] <= s[3]
        @test s[1] + s[2] + s[3] == d
    end

    states_f = collect(momentum_states(3, Ncut=2, d=d, particle_type=:fermion))
    for s in states_f
        @test s[1] < s[2] < s[3]
        @test s[1] + s[2] + s[3] == d
    end
end

@testset "惰性迭代特性" begin
    d = M(0, 0, 0)
    ch = momentum_states(2, Ncut=2, d=d)
    # 取前 3 个态验证不会一次性生成全部
    first_three = []
    for (i, state) in enumerate(ch)
        push!(first_three, state)
        i >= 3 && break
    end
    @test length(first_three) == 3
end

@testset "count_momentum_states" begin
    d = M(0, 0, 0)
    cnt = count_momentum_states(2, Ncut=2, d=d)
    @test cnt == length(collect(momentum_states(2, Ncut=2, d=d)))

    cnt_b = count_momentum_states(2, Ncut=3, d=d, particle_type=:boson)
    @test cnt_b == length(collect(momentum_states(2, Ncut=3, d=d, particle_type=:boson)))

    cnt_f = count_momentum_states(2, Ncut=3, d=d, particle_type=:fermion)
    @test cnt_f == length(collect(momentum_states(2, Ncut=3, d=d, particle_type=:fermion)))
end

@testset "输入格式兼容性" begin
    # NTuple 输入
    s1 = collect(momentum_states(2, Ncut=1, d=(0, 0, 0)))
    @test length(s1) > 0

    # Vector 输入
    s2 = collect(momentum_states(2, Ncut=1, d=[0, 0, 0]))
    @test length(s2) > 0

    # SVector 输入
    s3 = collect(momentum_states(2, Ncut=1, d=M(0, 0, 0)))
    @test length(s3) > 0

    # 预设常量 D000
    s4 = collect(momentum_states(2, Ncut=1, d=D000))
    @test length(s4) > 0

    @test length(s1) == length(s2) == length(s3) == length(s4)
end

@testset "错误处理" begin
    @test_throws ArgumentError momentum_states(2, Ncut=1, d="invalid")
    @test_throws ArgumentError momentum_states(2, Ncut=1, particle_type=:electron)
end

@testset "SymmetryGroup — O_h 群" begin
    @test length(O_h) == 48

    # 第 1 个是恒等元素
    I3 = SMatrix{3,3,Int}(1,0,0, 0,1,0, 0,0,1)
    @test O_h[1] == I3
    @test O_h[1] * M(1, 0, 0) == M(1, 0, 0)
    @test O_h[1] * M(0, 1, 0) == M(0, 1, 0)
    @test O_h[1] * M(0, 0, 1) == M(0, 0, 1)

    # 第 25 个是反演 (-I)
    @test O_h[25] == -SMatrix{3,3,Int}(1,0,0, 0,1,0, 0,0,1)
    @test O_h[25] * M(1, 1, 1) == M(-1, -1, -1)

    # 前 24 个是纯旋转（det = +1），后 24 个是旋转×反演（det = -1）
    for i in 1:24
        @test det(O_h[i]) == 1
        @test det(O_h[i+24]) == -1
    end

    # 所有矩阵行列式为 ±1
    for g in O_h
        @test abs(det(g)) == 1
    end

    # 所有矩阵将整数映射为整数（已是 SMatrix{3,3,Int} 构造）
    for g in O_h
        n = M(1, 2, 3)
        result = g * n
        @test result isa M
    end
end

@testset "SymmetryGroup — 子群" begin
    @test length(C4v) == 8
    @test length(C2v) == 4
    @test length(C3v) == 6

    # 子群元素必须全部在 O_h 中
    for g in C4v
        @test g in O_h
    end
    for g in C2v
        @test g in O_h
    end
    for g in C3v
        @test g in O_h
    end

    # 子群必须闭包（群性质基本验证）
    for sub in [C4v, C2v, C3v]
        for g1 in sub, g2 in sub
            product = g1 * g2
            # 至少模长保持（O(N)性质）
            v = M(1, 2, -3)
            @test sum(abs2, product * v) == sum(abs2, v)
        end
    end
end

@testset "SymmetryGroup — group_elements" begin
    @test length(group_elements(:Oh)) == 48
    @test length(group_elements(:C4v)) == 8
    @test length(group_elements(:C2v)) == 4
    @test length(group_elements(:C3v)) == 6

    @test group_elements(:Oh) === O_h
    @test group_elements(:C4v) === C4v
    @test group_elements(:C2v) === C2v
    @test group_elements(:C3v) === C3v

    @test_throws ArgumentError group_elements(:invalid)
end

@testset "SymmetryGroup — group_for_momentum" begin
    # (0,0,0) → O_h
    els, name = group_for_momentum(M(0, 0, 0))
    @test name == :Oh
    @test length(els) == 48

    # (0,0,1) → C4v
    els, name = group_for_momentum(M(0, 0, 1))
    @test name == :C4v
    @test length(els) == 8

    # (0,1,1) → C2v
    els, name = group_for_momentum(M(0, 1, 1))
    @test name == :C2v
    @test length(els) == 4

    # (1,1,1) → C3v
    els, name = group_for_momentum(M(1, 1, 1))
    @test name == :C3v
    @test length(els) == 6

    # 负号不影响群
    els2, name2 = group_for_momentum(M(0, 0, -1))
    @test name2 == :C4v
    @test length(els2) == 8

    # Tuple/Vector 输入
    els3, name3 = group_for_momentum((1, 1, 1))
    @test name3 == :C3v
    @test length(els3) == 6

    @test_throws ArgumentError group_for_momentum(M(1, 2, 3))
end

@testset "SymmetryGroup — apply_transform" begin
    # 恒等变换
    n = M(1, 2, 3)
    @test apply_transform(O_h[1], n) == n

    # 反演
    @test apply_transform(O_h[25], n) == -n

    # 多粒子态
    state = (M(1,0,0), M(0,1,0), M(0,0,1))
    transformed = apply_transform(O_h[2], state)
    @test length(transformed) == 3
    @test transformed[1] == O_h[2] * state[1]
    @test transformed[2] == O_h[2] * state[2]
    @test transformed[3] == O_h[2] * state[3]
end

@testset "SymmetryGroup — 与预设动量对应" begin
    # D000 → O_h
    @test group_for_momentum(D000)[2] == :Oh
    # D001 → C4v
    @test group_for_momentum(D001)[2] == :C4v
    # D011 → C2v
    @test group_for_momentum(D011)[2] == :C2v
    # D111 → C3v
    @test group_for_momentum(D111)[2] == :C3v
end

@testset "轨道分解 — group_orbit 基本性质" begin
    # 静止系二粒子玻色子，Ncut=2
    rep = (M(0,0,0), M(0,0,0))
    orbit = group_orbit(rep)
    @test length(orbit) == 1  # 恒等轨道
    @test orbit == [rep]

    # (0,0,1),(0,0,-1) 在 O_h 下的轨道
    rep2 = (M(0,0,1), M(0,0,-1))
    orbit2 = group_orbit(rep2)
    @test length(orbit2) >= 1
    @test rep2 in orbit2

    # 轨道中所有态总动量守恒
    for state in orbit2
        @test state[1] + state[2] == M(0, 0, 0)
    end
end

@testset "轨道分解 — boson 重排序" begin
    # 选一个代表，所有轨道态必须是排好序的
    reps = find_representatives(2, Ncut=2, particle_type=:boson)
    for rep in reps
        orbit = group_orbit(rep, particle_type=:boson)
        for state in orbit
            @test state[1] <= state[2]  # 玻色子排序
        end
    end
end

@testset "轨道分解 — fermion 重排序" begin
    reps = find_representatives(2, Ncut=3, particle_type=:fermion)
    for rep in reps
        orbit = group_orbit(rep, particle_type=:fermion)
        for state in orbit
            @test state[1] < state[2]  # 费米子严格排序
        end
    end
end

@testset "轨道分解 — 覆盖完整性" begin
    # 所有态的轨道并集 = 完整动量空间
    for pt in [:distinguishable, :boson, :fermion]
        all_states = collect(momentum_states(2, Ncut=2, particle_type=pt))
        all_set = Set(all_states)

        reps = find_representatives(2, Ncut=2, particle_type=pt)
        covered = Set{NTuple{2, M}}()
        for rep in reps
            orbit = group_orbit(rep, particle_type=pt)
            union!(covered, orbit)
        end

        @test covered == all_set
        @test length(covered) == length(all_states)
    end
end

@testset "轨道分解 — 轨道互不相交" begin
    for pt in [:distinguishable, :boson, :fermion]
        reps = find_representatives(2, Ncut=2, particle_type=pt)
        orbits = [Set(group_orbit(rep, particle_type=pt)) for rep in reps]

        # 每对轨道交集为空
        for i in 1:length(orbits), j in i+1:length(orbits)
            @test isempty(intersect(orbits[i], orbits[j]))
        end
    end
end

@testset "轨道分解 — 代表是最小态" begin
    for pt in [:distinguishable, :boson, :fermion]
        reps = find_representatives(2, Ncut=2, particle_type=pt)
        for rep in reps
            orbit = group_orbit(rep, particle_type=pt)
            # 代表必须是轨道中字典序最小的
            @test sort(collect(orbit))[1] == rep
        end
    end
end

@testset "轨道分解 — 用户文档例子验证" begin
    # N=2, d=(0,0,0), Ncut=2, boson
    # 用户的三个代表对（不计排序）：{(0,0,0),(0,0,0)}, {(0,0,1),(0,0,-1)}, {(0,1,1),(0,-1,-1)}
    reps = find_representatives(2, Ncut=2, particle_type=:boson)

    # 收集所有轨道中的所有态
    all_orbit_states = Set{NTuple{2, M}}()
    for rep in reps
        union!(all_orbit_states, group_orbit(rep, particle_type=:boson))
    end

    # 验证三对关键态都在完整轨道并集中
    pair1 = (M(0,0,0), M(0,0,0))
    @test pair1 in all_orbit_states

    # (0,0,1),(0,0,-1) 的对，排序后检查
    pair2 = (M(0,0,-1), M(0,0,1))
    @test pair2 in all_orbit_states

    # (0,1,1),(0,-1,-1) 的对，排序后检查
    pair3 = (M(0,-1,-1), M(0,1,1))
    @test pair3 in all_orbit_states

    # 代表态总数 = 轨道数
    covered = 0
    for rep in reps
        covered += length(group_orbit(rep, particle_type=:boson))
    end
    total = count_momentum_states(2, Ncut=2, particle_type=:boson)
    @test covered == total
end

@testset "轨道分解 — 不同总动量的群" begin
    # D001 → C4v（8 个群元）
    reps = find_representatives(2, Ncut=2, particle_type=:boson, d=D001)
    @test length(reps) > 0
    for rep in reps
        orbit = group_orbit(rep, d=D001, particle_type=:boson)
        # C4v 有 8 个群元，轨道大小不超过 8
        @test length(orbit) <= 8
        # 总动量依然守恒
        for state in orbit
            @test state[1] + state[2] == D001
        end
    end

    # D011 → C2v（4 个群元）
    reps2 = find_representatives(2, Ncut=2, particle_type=:boson, d=D011)
    for rep in reps2
        orbit = group_orbit(rep, d=D011, particle_type=:boson)
        @test length(orbit) <= 4
    end
end

@testset "多物种 — 与单物种向后兼容" begin
    # species=[3] 应与单物种 boson N=3 完全一致
    old = collect(momentum_states(3, Ncut=2, particle_type=:boson))
    new = collect(momentum_states(3, Ncut=2, species=[3], particle_types=[:boson]))
    @test length(old) == length(new)
    @test Set(old) == Set(new)
end

@testset "多物种 — [1,2] boson 排序约束" begin
    # 物种A: 1 boson, 物种B: 2 bosons
    # 仅物种B内部有排序约束
    states = collect(momentum_states(3, Ncut=2, species=[1,2], particle_types=[:boson, :boson]))
    @test length(states) > 0
    for s in states
        # 物种B内部（粒子2,3）需排序
        @test s[2] <= s[3]
    end
    # 应有更多态：粒子1与粒子2可交换
    states_single = collect(momentum_states(3, Ncut=2, particle_type=:boson))
    @test length(states) >= length(states_single)
end

@testset "多物种 — 计数一致性" begin
    for (spec, types) in [
        ([1, 2], [:boson, :boson]),
        ([2, 1], [:fermion, :boson]),
        ([1, 1, 1], [:distinguishable, :boson, :fermion]),
        ([2, 2], [:boson, :boson]),
    ]
        n = sum(spec)
        cnt = count_momentum_states(n, Ncut=2, species=spec, particle_types=types)
        collected = length(collect(momentum_states(n, Ncut=2, species=spec, particle_types=types)))
        @test cnt == collected
    end
end

@testset "多物种 — 总动量守恒" begin
    for (spec, types) in [
        ([1, 2], [:boson, :boson]),
        ([2, 1], [:fermion, :distinguishable]),
        ([1, 1, 1], [:boson, :boson, :boson]),
    ]
        n = sum(spec)
        for state in momentum_states(n, Ncut=2, species=spec, particle_types=types)
            total = sum(state)
            @test total == M(0, 0, 0)
        end
    end
end

@testset "多物种 — 粒子交换" begin
    # [1,2] boson: 物种B内交换不产生新态，但物种A与B交换产生新态
    states = collect(momentum_states(3, Ncut=2, species=[1,2], particle_types=[:boson, :boson]))
    # 查找：粒子1与粒子2交换后是否出现不同态
    found_cross_species = false
    for s in states
        swapped = (s[2], s[1], s[3])  # 交换物种A(粒子1)和物种B的第一个(粒子2)
        if s[1] != s[2] && !(swapped in states)  # 不对，我们需要检查...
            # 实际上交换不同物种的粒子不应产生合法态，因为排序约束不同
        end
    end
    # 简单验证：存在某个态 s[1] > s[2]（这在单物种boson中不可能）
    cross = any(s -> isless(s[2], s[1]), states)
    @test cross  # 多物种允许粒子1 > 粒子2
end

@testset "多物种 — fermion 跨物种" begin
    # [2,1] fermion+distinguishable: 物种A内2个费米子严格无重复
    states = collect(momentum_states(3, Ncut=3, species=[2,1], particle_types=[:fermion, :distinguishable]))
    for s in states
        # 物种A内部（粒子1,2）严格递增
        @test s[1] < s[2]
        # 粒子3无约束（可区分）
    end
end

@testset "多物种 — 轨道排序按物种块" begin
    # [1,2] boson 的轨道：群作用后仅物种B内部重排
    rep = (M(0,0,1), M(0,0,0), M(0,0,-1))  # 物种A=粒子1, 物种B=粒子2,3
    orbit = group_orbit(rep, species=[1,2], particle_types=[:boson, :boson])
    for state in orbit
        # 物种B内部排序
        @test state[2] <= state[3]
        # 总动量守恒
        @test state[1] + state[2] + state[3] == M(0, 0, 0)
    end
end

@testset "多物种 — 轨道覆盖完整性" begin
    for (spec, types) in [
        ([1, 2], [:boson, :boson]),
        ([2, 1], [:fermion, :boson]),
    ]
        n = sum(spec)
        all_states = collect(momentum_states(n, Ncut=2, species=spec, particle_types=types))
        all_set = Set(all_states)

        reps = find_representatives(n, Ncut=2, species=spec, particle_types=types)
        covered = Set{NTuple{n, M}}()
        for rep in reps
            orb = group_orbit(rep, species=spec, particle_types=types)
            union!(covered, orb)
        end

        @test covered == all_set
        @test length(covered) == length(all_states)
    end
end

@testset "多物种 — 轨道互不相交" begin
    for (spec, types) in [
        ([1, 2], [:boson, :boson]),
        ([2, 1], [:fermion, :boson]),
    ]
        n = sum(spec)
        reps = find_representatives(n, Ncut=2, species=spec, particle_types=types)
        orbits = [Set(group_orbit(rep, species=spec, particle_types=types)) for rep in reps]
        for i in 1:length(orbits), j in i+1:length(orbits)
            @test isempty(intersect(orbits[i], orbits[j]))
        end
    end
end

@testset "多物种 — 错误处理" begin
    # species 和不为 N
    @test_throws ArgumentError momentum_states(3, Ncut=1, species=[1, 1], particle_types=[:boson, :boson])
    # species 和 types 长度不匹配
    @test_throws ArgumentError momentum_states(3, Ncut=1, species=[1, 2], particle_types=[:boson])
    # 非法 particle_type
    @test_throws ArgumentError momentum_states(3, Ncut=1, species=[1, 2], particle_types=[:boson, :electron])
    # species 不是整数向量
    @test_throws ArgumentError momentum_states(3, Ncut=1, species="invalid")
end

@testset "多物种 — particle_type 作为默认" begin
    # species=[1,2], 不传 particle_types, particle_type=:boson 应作为默认
    states1 = collect(momentum_states(3, Ncut=2, species=[1,2], particle_types=[:boson, :boson]))
    states2 = collect(momentum_states(3, Ncut=2, species=[1,2], particle_type=:boson))
    @test length(states1) == length(states2)
    @test Set(states1) == Set(states2)
end

@testset "SpinSpace — spin 0 全部粒子" begin
    rep = (M(1,0,0), M(0,1,0))
    h = helicity_representatives(rep, species=[1,1], particle_types=[:distinguishable, :distinguishable], spins=[0, 0])
    @test length(h) == 1
    @test h[1] == (0, 0)
end

@testset "SpinSpace — 整数自旋可能值" begin
    rep = (M(1,0,0),)
    h = helicity_representatives(rep, species=[1], particle_types=[:distinguishable], spins=[1])
    @test length(h) >= 1
    for λ in h
        @test all(x -> x in (-1, 0, 1), λ)
    end
end

@testset "SpinSpace — 半整数自旋可能值" begin
    rep = (M(1,0,0),)
    h = helicity_representatives(rep, species=[1], particle_types=[:distinguishable], spins=[1//2])
    for λ in h
        @test all(x -> x in (Rational{Int}(-1,2), Rational{Int}(1,2)), λ)
    end
end

@testset "SpinSpace — 等价关系正确性" begin
    # 两个 spin-1/2 可区分粒子，验证等价关系 {λ} ∼ -{λ}
    rep = (M(1,0,0), M(0,1,0))
    h = helicity_representatives(rep, species=[1,1], particle_types=[:distinguishable, :distinguishable], spins=[1//2, 1//2])
    # 原始 4 个配置模掉 (λ₁,λ₂) ∼ (-λ₁,-λ₂) 得到 2 个等价类
    @test length(h) == 2
    # 每个代表的负号版本不应在结果中
    h_set = Set(h)
    for λ in h
        neg = Tuple(-x for x in λ)
        if neg != λ
            @test !(neg in h_set)
        end
    end
end

@testset "SpinSpace — 全同玻色子螺旋度交换" begin
    # 两个全同 spin-1/2 玻色子，(λ₁,λ₂) ∼ (λ₂,λ₁) ∼ - (λ₁,λ₂)
    rep = (M(0,0,-1), M(0,0,1))
    h = helicity_representatives(rep, species=[2], particle_types=[:boson], spins=[1//2])
    h_set = Set(h)
    for λ in h
        swapped = (λ[2], λ[1])
        neg = Tuple(-x for x in λ)
        neg_swapped = Tuple(-x for x in swapped)
        # 所有等价态不应重复出现
        if swapped != λ
            @test !(swapped in h_set)
        end
        if neg != λ
            @test !(neg in h_set)
        end
        if neg_swapped != λ && neg_swapped != swapped && neg_swapped != neg
            @test !(neg_swapped in h_set)
        end
    end
end

@testset "SpinSpace — 重复动量置换" begin
    # 两个全同玻色子处于相同动量（重复动量），应有置换等价
    rep = (M(1,0,0), M(1,0,0))
    h = helicity_representatives(rep, species=[2], particle_types=[:boson], spins=[1//2])
    h_set = Set(h)
    for λ in h
        swapped = (λ[2], λ[1])
        if swapped != λ
            @test !(swapped in h_set)
        end
    end
end

@testset "SpinSpace — 零动量+非零自旋报错" begin
    @test_throws ArgumentError helicity_representatives(
        (M(0,0,0), M(1,0,0)), species=[1,1], particle_types=[:distinguishable, :distinguishable], spins=[1//2, 0])
    @test_throws ArgumentError helicity_representatives(
        (M(1,0,0), M(0,0,0)), species=[1,1], particle_types=[:distinguishable, :distinguishable], spins=[0, 1//2])
end

@testset "SpinSpace — 零动量+零自旋允许" begin
    # s=0 的粒子允许零动量
    h = helicity_representatives(
        (M(0,0,0),), species=[1], particle_types=[:boson], spins=[0])
    @test length(h) == 1
    @test h[1] == (0,)
end

@testset "SpinSpace — 多物种 helicity" begin
    # [1,2] boson, 物种A spin 0, 物种B spin 1/2
    rep = (M(1,0,0), M(0,1,0), M(0,0,1))
    h = helicity_representatives(rep, species=[1,2], particle_types=[:boson, :boson], spins=[0, 1//2])
    @test length(h) > 0
    for λ in h
        @test λ[1] == 0  # 物种A spin 0
        @test λ[2] in (Rational{Int}(-1,2), Rational{Int}(1,2))
        @test λ[3] in (Rational{Int}(-1,2), Rational{Int}(1,2))
    end
end

@testset "SpinSpace — 不同总动量使用不同群" begin
    # D001 → C4v (8 elements) vs D000 → O_h (48 elements)
    # Per 大小不同可能导致不同的等价关系
    rep = (M(0,0,1), M(0,0,-1))
    h_oh = helicity_representatives(rep, species=[2], particle_types=[:boson], spins=[1//2], d=D000)
    h_c4v = helicity_representatives(rep, species=[2], particle_types=[:boson], spins=[1//2], d=D001)
    @test length(h_oh) >= 1
    @test length(h_c4v) >= 1
end

@testset "SpinSpace — 字典序排列" begin
    h = helicity_representatives(
        (M(1,0,0), M(0,1,0)), species=[1,1], particle_types=[:distinguishable, :distinguishable], spins=[1, 1])
    for i in 1:length(h)-1
        @test isless(h[i], h[i+1]) || h[i] == h[i+1]
    end
end

@testset "SpinSpace — 错误输入" begin
    rep = (M(1,0,0), M(0,1,0))
    @test_throws ArgumentError helicity_representatives(rep, species=[1], particle_types=[:distinguishable], spins=[0])
    @test_throws ArgumentError helicity_representatives(rep, species=[1,1], particle_types=[:distinguishable], spins=[0])
    @test_throws ArgumentError helicity_representatives(rep, species=[1,1], particle_types=[:distinguishable, :distinguishable], spins=[0, 0, 0])
end

# ============ Isospin 参数测试 ============

@testset "SpinSpace — 无同位旋参数时行为不变" begin
    rep = (M(1,0,0), M(0,1,0))
    h1 = helicity_representatives(rep, species=[1,1],
        particle_types=[:distinguishable, :distinguishable], spins=[1//2, 1//2])
    h2 = helicity_representatives(rep, species=[1,1],
        particle_types=[:distinguishable, :distinguishable], spins=[1//2, 1//2],
        isospins=nothing, subsystem_isospins=nothing, total_isospin=nothing)
    @test h1 == h2
end

@testset "SpinSpace — 同位旋参数必须全提供" begin
    rep = (M(1,0,0), M(0,1,0))
    @test_throws ArgumentError helicity_representatives(rep, species=[1,1],
        particle_types=[:distinguishable, :distinguishable], spins=[1//2, 1//2],
        isospins=[1//2, 1//2])
    @test_throws ArgumentError helicity_representatives(rep, species=[1,1],
        particle_types=[:distinguishable, :distinguishable], spins=[1//2, 1//2],
        subsystem_isospins=[1//2, 1//2])
    @test_throws ArgumentError helicity_representatives(rep, species=[1,1],
        particle_types=[:distinguishable, :distinguishable], spins=[1//2, 1//2],
        total_isospin=0)
end

@testset "SpinSpace — 同位旋长度必须匹配 species" begin
    rep = (M(1,0,0), M(0,1,0), M(0,0,1))
    @test_throws ArgumentError helicity_representatives(rep, species=[1,2],
        particle_types=[:boson, :boson], spins=[1//2, 1//2],
        isospins=[1//2], subsystem_isospins=[1//2, 1//2], total_isospin=0)
    @test_throws ArgumentError helicity_representatives(rep, species=[1,2],
        particle_types=[:boson, :boson], spins=[1//2, 1//2],
        isospins=[1//2, 1//2], subsystem_isospins=[1//2], total_isospin=0)
end

@testset "SpinSpace — 同位旋值必须非负整数或半整数" begin
    rep = (M(1,0,0),)
    @test_throws ArgumentError helicity_representatives(rep, species=[1],
        particle_types=[:boson], spins=[0],
        isospins=[-1//2], subsystem_isospins=[1//2], total_isospin=0)
    @test_throws ArgumentError helicity_representatives(rep, species=[1],
        particle_types=[:boson], spins=[0],
        isospins=[1//2], subsystem_isospins=[-1], total_isospin=0)
    @test_throws ArgumentError helicity_representatives(rep, species=[1],
        particle_types=[:boson], spins=[0],
        isospins=[1//2], subsystem_isospins=[1//2], total_isospin=-1//2)
    @test_throws ArgumentError helicity_representatives(rep, species=[1],
        particle_types=[:boson], spins=[0],
        isospins=[0.3], subsystem_isospins=[1//2], total_isospin=0)
end

@testset "SpinSpace — 合法同位旋参数正常返回" begin
    rep = (M(1,0,0), M(0,1,0))
    # 两个可区分核子 (I=1/2)，各自耦合到 I=1/2，总 I=0
    h = helicity_representatives(rep, species=[1,1],
        particle_types=[:distinguishable, :distinguishable], spins=[1//2, 1//2],
        isospins=[1//2, 1//2], subsystem_isospins=[1//2, 1//2], total_isospin=0)
    @test length(h) > 0

    # 多物种 + 同位旋
    rep2 = (M(1,0,0), M(0,1,0), M(0,0,1))
    h2 = helicity_representatives(rep2, species=[1,2],
        particle_types=[:boson, :boson], spins=[0, 1//2],
        isospins=[0, 1//2], subsystem_isospins=[0, 1], total_isospin=1)
    @test length(h2) > 0
    for λ in h2
        @test λ[1] == 0  # spin 0 物种
    end
end

@testset "SpinSpace — 整数同位旋" begin
    rep = (M(1,0,0),)
    h = helicity_representatives(rep, species=[1],
        particle_types=[:boson], spins=[0],
        isospins=[1], subsystem_isospins=[1], total_isospin=1)
    @test length(h) == 1
end

# ============ 耦合道本征值归属性测试 ============

@testset "DD + D*D* 耦合道 (I=1, κ=[2])" begin
    include("test_dd_dstardstar.jl")
end

@testset "πN → πN 单道 (双覆盖群)" begin
    include("test_piN.jl")
end
