# 从密度矩阵导数到 VQS 参数演化（Phase 5）

本阶段实现论文 Eq. (28)/(29) 的 M、V，并按已批准方案演化六个参数。没有读取 Exact 数据来求参数。论文依据沿用[第11页公式核对](../papers/phase4_derivatives_source_zh.md)、[第20页线路约定](../papers/figure4_ansatz_source_zh.md)与[第20–21页模型记录](../papers/figure5_exact_source_zh.md)。Forward Euler 和下述 SVD 阈值是本项目的数值选择，不声称是作者公开的具体实现。

## 1. 约定与最小化问题

系统空间为 $\mathbb C^2\otimes\mathbb C^2$，基底为 $|00\rangle,|01\rangle,|10\rangle,|11\rangle$；辅助位先于系统位，线路顺序 $(a_0,a_1,s_0,s_1)$。令 $\hbar=1$，以横场系数1定义无量纲时间。六个 $\theta_i$ 是实弧度参数，初值均为0。约化密度矩阵 $\rho(\theta)$ 与导数 $D_i=\partial_i\rho$ 均为4×4复矩阵。

动力学生成元采用已验收模型：

$$
H=X\otimes I+I\otimes X+\tfrac14Z\otimes Z,\qquad
c_0=|0\rangle\langle1|\otimes I,\quad c_1=I\otimes|0\rangle\langle1|,
$$

$$
\mathcal L(\rho)=-i[H,\rho]+\sum_{j=0}^1
\left(c_j\rho c_j^\dagger-\tfrac12\{c_j^\dagger c_j,\rho\}\right).
$$

参数变化只能沿线路提供的切空间产生 $\dot\rho=\sum_i D_i\dot\theta_i$。McLachlan 方法在当前态处最小化 Hilbert–Schmidt 范数平方：

$$
F(x)=\left\|\sum_i D_i x_i-\mathcal L(\rho)\right\|_F^2
=\sum_{ij}x_iM_{ij}x_j-2\sum_i x_iV_i+\|\mathcal L(\rho)\|_F^2,
$$

$$
M_{ij}=\operatorname{Re}\operatorname{Tr}(D_i^\dagger D_j),\qquad
V_i=\operatorname{Re}\operatorname{Tr}[D_i^\dagger\mathcal L(\rho)].
$$

对实数 $x_k$ 求导，得到 $2\sum_jM_{kj}x_j-2V_k=0$，所以 $Mx=V$。由于 $M$ 是 Gram 矩阵，它对称、半正定，却不一定可逆。这里的度量来自系统密度矩阵，不能换成四位纯态导数的内积。

## 2. 公式如何对应代码

- `mclachlan.lindblad_rhs` 在当前 VQS 密度矩阵处计算生成元；Hamiltonian 与全部耗散项都有保留。
- `raw_equations` 的 `einsum("ist,jst->ij", conj(D), D)` 是对矩阵元素求和：$\operatorname{Tr}(A^\dagger B)=\sum_{st}A_{st}^*B_{st}$。第一个轴 i 是参数轴，后两个是系统矩阵轴。它不是省略了共轭转置。
- `evaluate_equations` 完成 JAX 自动微分后才转到 NumPy。它先检查 M/V 虚部、M 对称性及最小特征值，再取实部；容差1e-12，不通过取实部掩盖错误。
- `solve_svd` 返回参数速度和谱/残差；`vqs.evolve_vqs` 做参数 Forward Euler，每次从线路重新构造密度矩阵。

## 3. 全零初参的解析结果

初态为 $|00\rangle\langle00|$，辅助位为 $|++\rangle$。Phase 4 已验证 $D_3=2D_0$、$D_4=2D_1$、$D_2=D_5=0$。M 只有索引 (0,3) 和 (1,4) 的两块非零，每块为

$$
\begin{pmatrix}1/2&1\\1&2\end{pmatrix}.
$$

因此理想奇异值为 $(2.5,2.5,0,0,0,0)$，秩2；$V=(1,1,0,2,2,0)$。其中一块要求 $x_0+2x_3=2$。使 $x_0^2+x_3^2$ 最小的解沿 $(1,2)$，为 $(0.4,0.8)$；另一块相同。因此

$$
\dot\theta(0)=(0.4,0.4,0,0.8,0.8,0).
$$

初点耗散项为0，初点切空间能够表示该时刻的 Hamiltonian 生成元。该事实只适用于初点，不保证后续时刻也能完全表示动力学。实际浮点计算的两个近零奇异值约1.17e-16；没有把它们当作可逆方向。

## 4. SVD 的阈值、条件数与残差

若 $M=U\Sigma W^T$，保留 $s_k>\tau$ 的方向，令

$$
\tau=6\epsilon_{64}s_{\max},\quad
x=W\Sigma_\tau^+U^TV,\qquad \epsilon_{64}\simeq2.22\times10^{-16}.
$$

这给出保留子空间中的最小范数最小二乘解。维数×机器精度的相对阈值参考 [NumPy 1.26 `lstsq` 文档](https://numpy.org/doc/1.26/reference/generated/numpy.linalg.lstsq.html)；[SVD 文档](https://numpy.org/doc/1.26/reference/generated/numpy.linalg.svd.html)说明矩阵分解与排序。此处显式实现阈值，避免依赖默认参数变化。初点绝对阈值约3.33e-15。没有 Tikhonov 项或特殊首步处理。

完整条件数在数值秩小于6时记为 infinity；保留谱条件数为 $s_{\max}/s_{\min,\mathrm{kept}}$。初点前者无穷、后者1，两者含义不同。NPZ 中的条件数 infinity 是明确允许的诊断标记，其余计算数组必须有限；JSON 初点用文字说明无穷，避免非标准 JSON Infinity。

线性残差 $\|Mx-V\|_2$ 检查求解是否满足方程；切空间残差 $\|\sum_iD_ix_i-\mathcal L(\rho)\|_F$ 检查当前 ansatz 能表示多少物理生成元。即使前者很小，后者也可能非零。高条件数意味着参数速度可能对舍入敏感，不能只凭线性残差断言长期准确。

## 5. 参数更新、保存与检查

基线严格使用

$$
\theta_{n+1}=\theta_n+0.01\dot\theta_n,\quad n=0,\ldots,999.
$$

所有 `rho[n]`、`z0[n]`、M/V、速度与诊断都在 `theta[n]`、$t_n=0.01n$ 处计算。`parameter_step` 只有1000项；最后一个时间点保存诊断但不继续更新。`rho[n+1]` 来自线路，绝不使用 $\rho_n+dt\mathcal L(\rho_n)$ 代替。没有归一化、厄米修复、特征值裁剪或改变初态。

独立验证不是复述同一张量表达式：使用 Phase 3 的16×16稠密门、Phase 4 的门乘积法则导数、QuTiP Liouvillian 和显式 trace 循环核对17组参数。还把复切向量拆成实/虚两块设计矩阵 B（32×6），直接解密度空间最小二乘，避免形成 $B^TB$ 来对照参数解。相应 B 相对阈值是 $\sqrt{6\epsilon_{64}}$。

所有1001个状态都检查迹、厄米性和最小特征值，物理容差1e-10；更新与观测量对齐容差1e-12，并对11个覆盖全区间的样本用独立门计算重建密度矩阵。保存文件读回逐元素一致。

## 6. 实际结果与解释范围

证据：[results/vqs_20261014_161232_948193](../results/vqs_20261014_161232_948193/)，包括基线、短时诊断与版本/代码/数据哈希记录。

- 全程迹最大误差1.110e-15、厄米误差0、最小密度特征值0；初点秩2，之后1000点数值秩6。
- M/V 与独立参考最大元素差分别1.443e-15/1.554e-15（容差1e-12）；JIT 与未编译计算差0。
- 17组参数解与直接密度最小二乘的最大元素差1.224e-10（容差1e-9）。
- 全程线性残差最大5.759e-15；切空间残差最大0.0315028；最大保留谱条件数3.173e8，最大参数步长范数0.0314884；$Z_0(10)=0.1218784663$。
- 在 $0\le t\le0.2$，阈值×10/×100均未改变秩或结果。这不证明全时间阈值不敏感。
- 在同一短区间，dt=.01与.005的 Z0 最大差1.619e-4，.005与.0025的最大差8.365e-5。相邻差减小是短时数值观察，不是全时间收敛证明；更小步长数据不替代基线。

本阶段完成可独立运行的 VQS 数据，尚未计算与 Exact 的全点差值、判断论文1e-2目标或绘制 Figure 5；这些属于待批准 Phase 6。

## 7. 使用

完整命令见[英文 README](../README.md)。运行 `scripts/run_vqs.py` 会先验证，再创建新的 `results/vqs_<时间戳>/`，保留历史输出。`vqs.npz` 保存基线；`short_diagnostics.npz` 用不同前缀保存阈值和步长诊断。两者不能混作同一条曲线。没有新增依赖，使用已固定的项目虚拟环境。
