# NPHFforFVE.jl

Non-Perturbative Hamiltonian Framework for Finite Volume Energy spectra.

有限体积哈密顿量方法 — 格点能谱计算框架。

## 功能

- N=2 多粒子散射态有限体积能谱计算
- 支持可区分粒子、全同玻色子、全同费米子
- 静止系 (O_h / 2O_h) 与运动系 (C4v2 / C2v2 / C3v2)
- 多物种同位旋分解 (S_N 置换群)
- 螺旋度基旋量投影
- 自动模板生成

## 安装

```julia
import Pkg
Pkg.add(url="https://github.com/yukangUCAS/NPHFforFVE.jl")
```

## 快速开始

```julia
using NPHFforFVE

# 1. 定义 Fock 道
ch_piN = FockChannel("piN", [1, 1], [:boson, :fermion],
                     [139.57, 938.92], [0//1, 1//2], [1//1, 1//2],
                     [1.0, 1.0], relativistic)

# 2. 创建项目
proj = Project(Momentum(0,0,0), 1//2, [ch_piN], [20])

# 3. 添加组态 (L, a, 不可约表示, 能级数)
add_config!(proj, 48, 0.1, ["G1-", "G2-"], [5, 5])

# 4. 计算
result = compute!(proj, my_V, params)

# 5. 提取能级
result[1]["G1-"]   # G1- 道能级 [MeV]
```

## 生成势能模板

```julia
generate_potential_template("potential_defs.jl", d=Momentum(0,0,1), I=1//2, channels=[ch_piN])
```

## 测试

```julia
import Pkg; Pkg.test()
```

## 许可

MIT License
