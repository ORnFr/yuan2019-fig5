# 密度矩阵 Jacobian：公式、轴序与验证

日期：2026-10-14。Phase 4只计算系统密度矩阵对线路参数的导数，没有计算McLachlan方程或时间演化。原文位置与软件方法的实际核对见 [来源记录](../papers/phase4_derivatives_source_zh.md)。

## 从量子态变化到参数导数

空间、基底和Figure 4线路沿用 [purification说明](ansatz_purification_zh.md)：整体 $\mathcal H_A\otimes\mathcal H_S$ 为16维，辅助/系统各4维；位序(a0,a1,s0,s1)，系统基底 $|00\rangle,|01\rangle,|10\rangle,|11\rangle$。初始六参数均为零，系统初态 $\rho_0=|00\rangle\langle00|$。没有空间边界条件或维数截断。参数是六个有限实数、无量纲弧度；本阶段没有物理时间或积分步长。

给参数一个小改变量，密度矩阵的第一阶变化是

$$
\rho_S(\theta+\delta\theta)=\rho_S(\theta)+\sum_{i=0}^5D_i(\theta)\delta\theta_i+O(\|\delta\theta\|^2),
\qquad D_i=\frac{\partial\rho_S}{\partial\theta_i}.
$$

这里每个 $D_i$ 是4×4复矩阵，表示一个参数变化时整个系统密度矩阵的变化方向。六个矩阵合在一起就是Jacobian，保存为shape `(6,4,4)`。第一阶展开是小改变量近似，$D_i$ 的定义本身是严格导数，不是Euler演化。

## 共轭关系与自动微分

由 $\rho_S=\operatorname{Tr}_A(|\Psi\rangle\langle\Psi|)$，对实参数严格求导可得

$$
D_i=\operatorname{Tr}_A\left[|\partial_i\Psi\rangle\langle\Psi|+|\Psi\rangle\langle\partial_i\Psi|\right].
$$

关键是 $\partial_i(\Psi^*)=(\partial_i\Psi)^*$，这里用到参数是实数。核心实现并不先构造整个16×16密度矩阵再求导，而是直接对已经验收的 `system_density(theta)` 求导；上式用于独立矩阵参考。rho中有共轭，不能通过把theta强行改成复数、设 `holomorphic=True` 来忽略它。

`density_jacobian(theta)` 采用

```python
real_jacobian = jax.jacfwd(lambda p: jnp.real(system_density(p)))(theta)
imaginary_jacobian = jax.jacfwd(lambda p: jnp.imag(system_density(p)))(theta)
D = jnp.moveaxis(real_jacobian + 1j * imaginary_jacobian, -1, 0)
```

`jacfwd` 是前向自动微分，逐输入方向传播导数；它不是用附近点做差。输入shape `(6,)`，输出rho为 `(4,4)`，所以JAX初始Jacobian轴序为 `(s,t,i)`、shape `(4,4,6)`。`moveaxis(...,-1,0)` 把参数轴搬到最前面，得到 `(i,s,t)`。计算过程保留JAX数组和float64/complex128；转换为NumPy只发生在求导结束后的独立检查与文件保存中。

库门角度是 `2*theta_i`，因此AD自动包含角度链式法则的2。对库的 $\exp(-i\phi X/2)$ 且 $\phi=2\theta$ 求导，结果是 $-iXR_X$，不能漏掉该因子或再额外乘一次2。

## 为什么导数应该厄米且迹为零

对每个实参数，$\rho_S^\dagger=\rho_S$ 且 $\operatorname{Tr}\rho_S=1$，所以逐参数求导得到

$$
D_i^\dagger=D_i,\qquad \operatorname{Tr}D_i=0.
$$

导数是切向量，不是密度矩阵：不要求迹为1，也不要求所有特征值非负。本项目检查未修复导数的厄米性和迹，不通过强制对称化或减去迹来掩盖错误。

## 零参数点的独立解析答案

零参数完整态为 $|++\rangle_A|00\rangle_S$。受控RX的生成元为辅助投影 $P_1=|1\rangle\langle1|$ 乘系统X；辅助态的 $\operatorname{Tr}(P_1|+\rangle\langle+|)=1/2$。对全态导数取部分迹，严格得到

$$
D_0(0)=-\frac i2[X_0,\rho_0],\qquad D_1(0)=-\frac i2[X_1,\rho_0].
$$

系统上的无控制旋转则给出

$$
D_3(0)=-i[X_0,\rho_0]=2D_0(0),\qquad
D_4(0)=-i[X_1,\rho_0]=2D_1(0).
$$

两个RZZ在零参数点的导数都是 $-i[Z_0Z_1,\rho_0]=0$，因为初态是ZZ本征态；它们此时只改变整体相位，所以

$$
D_2(0)=D_5(0)=0.
$$

例如 $D_0$ 的(0,2)元素为 $i/2$、(2,0)为 $-i/2$；$D_1$ 相应非零元素是(0,1)、(1,0)。六个导数的Frobenius范数为 $(1/\sqrt2,1/\sqrt2,0,\sqrt2,\sqrt2,0)$。不同的两对方向线性相关，因此零点只有两个独立第一阶系统密度方向。这个结论来自上述明确矩阵，而不是把小奇异值强行置零。没有为消除冗余而更改初始参数。后续McLachlan矩阵的实际谱、秩和求解策略仍属Phase 5，本次不计算。

## 独立通用参考：门乘积法则

`dense_state_and_derivatives` 用NumPy/SciPy矩阵验证：若当前一步为 $\Psi_{new}=G\Psi_{old}$，则

$$
\partial_i\Psi_{new}=G\partial_i\Psi_{old}+(\partial_iG)\Psi_{old}.
$$

只有门自身对应的参数才有第二项。RX导数 $-iXR_X$，RZZ导数 $-iZZR_{ZZ}$；受控门的control=0导数块是零、control=1块是RX导数。得到六个纯态导数后，按前面的两个outer-product项构造完整导数密度矩阵，并显式求和辅助对角块。这条参考不调用JAX求导、不用中心差分，也不复用正式部分迹的reshape/einsum。

## 有限差分与误差定义

有限差分只用于验收：

$$
D_i^{FD}(h)=\frac{\rho_S(\theta+h e_i)-\rho_S(\theta-h e_i)}{2h}.
$$

h是参数角度增量，不是物理时间步长。在光滑函数的精确算术Taylor展开中，中心差分的首个截断项为 $h^2\partial_i^3\rho_S/6$；实际浮点比较还含有限精度影响，不能只凭一组误差数值认定全部来源。

逐参数绝对误差定义为 $\max_{s,t}|D_i^{FD}-D_i^{AD}|$；相对误差定义为 $\|D_i^{FD}-D_i^{AD}\|_F/\|D_i^{dense}\|_F$。当独立参考范数不大于1e-12时，相对误差未定义，只检查绝对误差。NPZ的 `fd_relative_defined` 标记这些方向；未定义条目里的零仅为存储占位，不能解读为误差为零。

| h | 实际最大绝对误差 | 绝对容差 | 最大已定义相对误差 | 相对容差 |
| --- | --- | --- | --- | --- |
| 1e-3 | 6.667e-7 | 2e-6 | 6.667e-7 | 5e-6 |
| 1e-4 | 6.667e-9 | 2e-8 | 6.668e-9 | 5e-8 |
| 1e-5 | 6.667e-11 | 2e-9 | 7.708e-11 | 5e-8 |
| 1e-6 | 1.474e-10 | 2e-9 | 2.142e-10 | 5e-8 |

全部17组参数、六个方向和四个步长通过各自容差；12个近零参考方向未定义相对误差。最小h的误差没有继续下降，不能宣称这一范围中误差单调收敛或已经唯一确定误差来源。

## 实际交付与限制

AD与独立乘积法则参考的最大元素误差为5.551e-16（容差1e-12）；零点解析比较误差4.441e-16。导数迹最大误差2.238e-16、厄米Frobenius误差0。对零点和两个非零点，JVP方向导数与Jacobian收缩最大差2.220e-16，JIT与未编译输出差0；两者容差均为1e-12。这些AD一致性检查不替代独立数学参考。

运行与保存说明见README。数据及元数据位于 `results/derivatives_20261014_154702_356659/`，含17组参数、rho、两条路径的Jacobian `(17,6,4,4)` 和逐参数FD误差。随机种子2019只用于八组验证参数，未读取Exact数据拟合。合并测试22项通过，依赖未改变。

本阶段不包含M、V、伪逆、正则化或Euler步骤，未产生VQS轨迹。保留零参数起点并停止，等待用户验收后再讨论Phase 5。
