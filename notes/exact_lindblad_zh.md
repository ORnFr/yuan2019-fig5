# Exact 演化：从主方程到代码

日期：2026-10-13。模型和出处见 `papers/figure5_exact_source_zh.md`；以下是本项目对论文 Eq. (62) 的解释与实现，未引入变分近似。

## 模型、空间与约定

系统希尔伯特空间为 $\mathcal H=\mathbb C^2\otimes\mathbb C^2$，基底按 $|00\rangle,|01\rangle,|10\rangle,|11\rangle$ 排列。第 0 位是左侧张量因子；因此测量第 0 位时，$Z_0=Z\otimes I$，不是 $I\otimes Z$。初态 $\rho_0=|00\rangle\langle00|=\operatorname{diag}(1,0,0,0)$。没有辅助位、空间边界条件或截断维数近似。

采用 $\hbar=1$，Hamiltonian 的横场系数为 1，以此定义能量/频率标度。时间为无量纲时间；若恢复能量尺度 $E_0$，则 $t=E_0t_{physical}/\hbar$，这里的阻尼率 1 对应 $\gamma_{physical}\hbar/E_0=1$。

$$
H=X\otimes I+I\otimes X+\tfrac14 Z\otimes Z,
\quad X=\begin{pmatrix}0&1\\1&0\end{pmatrix},\quad Z=\begin{pmatrix}1&0\\0&-1\end{pmatrix}.
$$

矩阵形式为

$$
H=\begin{pmatrix}
1/4&1&1&0\\1&-1/4&0&1\\1&0&-1/4&1\\0&1&1&1/4
\end{pmatrix}.
$$

`model.build_model()` 用 QuTiP `tensor` 构造，验证时另用上述显式 NumPy 矩阵逐项比较。`tensor(A,B)` 表示 $A\otimes B$。

## 从封闭体系到 Lindblad 主方程

已熟悉的封闭体系密度矩阵方程是 $\dot\rho=-i[H,\rho]$。在这个模型中，还要加入两个独立振幅阻尼通道：

$$
\dot\rho=\mathcal L(\rho)=-i[H,\rho]
+\sum_{j=0}^1\left[c_j\rho c_j^\dagger-\tfrac12(c_j^\dagger c_j\rho+\rho c_j^\dagger c_j)\right].
$$

$$
\sigma^-=|0\rangle\langle1|=\begin{pmatrix}0&1\\0&0\end{pmatrix},
\qquad c_0=\sigma^-\otimes I,\quad c_1=I\otimes\sigma^-.
$$

对任意阻尼率 $\gamma$，collapse operator 应为 $\sqrt\gamma\,\sigma^-$；此处 $\gamma=1$。式中的 $1/2$ 属于反对易项，不能再把 $c_j$ 乘 $1/\sqrt2$，否则会把人口衰减率改成 1/2。

代码使用 `basis(2,0) * basis(2,1).dag()`，直接对应 $|0\rangle\langle1|$，避免凭 QuTiP 的 `sigmam` 名称猜测计算基底方向。验证检查完整 collapse 矩阵，包含 $c_0|10\rangle=|00\rangle$、$c_1|01\rangle=|00\rangle$，以及它们对其他基矢的作用。

## 解析检查：单比特、无 Hamiltonian

令 $H=0$、$c=|0\rangle\langle1|$、$\rho(0)=|1\rangle\langle1|$。此时 $c^\dagger c=|1\rangle\langle1|$；$c\rho c^\dagger$ 的 11 元素为零，两个反对易项的 11 元素均为 $\rho_{11}$。因此

$$
\dot\rho_{11}=-\rho_{11},\qquad
\rho_{11}(t)=e^{-t},\qquad
\langle Z\rangle=\rho_{00}-\rho_{11}=1-2e^{-t}.
$$

`validation.check_analytic_damping()` 独立比较这个解析解，检查符号、方向和耗散率。本次最大人口误差为 `2.22e-16`，最大 Z 误差为 `3.33e-16`，声明容差为 `1e-9`。

## 求解与保存时间点

`exact.solve_lindblad()` 使用 QuTiP `mesolve` 的 DOP853 自适应积分。这个算法在内部选择步长；`0.01` 是要求输出的间隔，不是 Euler 步长。绝对/相对局部容差分别为 `1e-12` 和 `1e-10`，不等于保证了相同数值的全局误差。

`exact.evolve_exact()` 创建 `0.01 * np.arange(1001)`；`arange(1001)` 是整数编号 0…1000，所以包含初态和 t=10，一共 1001 点。每个 `state.full()` 取出对应时刻的 4×4 密度矩阵，`np.stack` 沿新的时间轴堆叠，得到 shape `(1001,4,4)`。

观测量通过

$$
z_0(t)=\operatorname{Re}\operatorname{Tr}[Z_0\rho(t)]
$$

计算。`np.einsum("ij,tji->t", Z0, rho)` 实现 $\sum_{ij}(Z_0)_{ij}\rho_{ji}(t)$，保留时间轴，输出 shape `(1001,)`。代码先检查虚部是否低于容差，再取实部；没有用取实部隐藏显著错误。

## 独立矩阵指数参考与物理检查

生成元对密度矩阵是线性的。将矩阵逐列展开为长度 16 的向量 $r=\mathrm{vec}_F(\rho)$，有

$$
\dot r=Ar,\qquad r(t)=e^{tA}r(0).
$$

`validation.reference_liouvillian()` 对 16 个矩阵单位逐个应用手写 NumPy 主方程，构造 $A$ 的各列，和 QuTiP 完整 Liouvillian 比较。随后用 SciPy `expm_multiply` 独立计算全时间网格，不调用 `mesolve`。代码中 `reshape(..., order="F")` 表示逐列展开；返回数组恢复成方阵时的转置是为了恢复这一顺序，不是部分迹。

物理检查使用未修复的状态：迹误差 $\max_t|\operatorname{Tr}\rho-1|$，厄米误差 $\max_t\|\rho-\rho^\dagger\|_F$，最小特征值。计算特征值时使用厄米部分仅作诊断，已经先检查原状态的厄米误差；不把对称化后的状态保存为演化结果。开放体系一般不守恒系统能量，本项目不把能量守恒作为这个耗散模型的验收条件。

本次全 1001 点的最大密度矩阵元素差为 `2.844e-15`，最大 Z0 差为 `2.276e-15`；比较容差 `1e-9`。迹误差 `1.110e-15`、厄米误差 `4.170e-16`、最小特征值 0，满足 `1e-10` 的物理容差。更严格 ODE 容差在本次输出网格上的结果与基线逐元素相同；不能据此把舍入量级的差异称为严格数学误差界。

`exact.npz` 保存 t、rho、z0；`run_metadata.json` 记录参数、单位、误差定义、容差、实际检查结果、版本和源文件哈希。随机种子 2019 只用于生成元的测试态，物理演化没有随机性。完整 7 项测试通过。论文图形采样坐标的推断另见论文记录，不影响这里定义的物理演化。
