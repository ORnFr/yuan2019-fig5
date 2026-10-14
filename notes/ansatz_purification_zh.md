# 从四位纯态到两位混态：Figure 4 与部分迹

日期：2026-10-14。原文位置与门源码核对见 [论文记录](../papers/figure4_ansatz_source_zh.md)。本说明将 Phase 3 的公式与英文代码对应；没有开始参数求导或时间演化。

## 空间、基底、参数和初态

整体空间为 $\mathcal H_A\otimes\mathcal H_S$，其中辅助空间和系统空间各为 $\mathbb C^2\otimes\mathbb C^2\cong\mathbb C^4$。位序固定为 $(a_0,a_1,s_0,s_1)=(0,1,2,3)$；完整基底为 $|a_0a_1s_0s_1\rangle$，最左位是最高有效位，向量下标为

$$
n=8a_0+4a_1+2s_0+s_1=4a+s,
\quad a=2a_0+a_1,\quad s=2s_0+s_1.
$$

完整纯态向量有16个复振幅；系统密度矩阵在 $|00\rangle,|01\rangle,|10\rangle,|11\rangle$ 基底中为4×4矩阵，与 Exact 模块的系统基底一致。初态为 $|0000\rangle$；输入参数为六个有限实数 $\theta=(\theta_0,\ldots,\theta_5)$，shape `(6,)`、float64，角度以弧度计且无量纲。本阶段没有物理时间、积分步长、截断维数或新的边界条件，也不需要为角度额外指定能量单位。

## 六参数线路及角度

`build_ansatz(theta)` 按下表从上到下执行；这是时间顺序。若写成矩阵乘积，最后执行的门在最左侧。

| 操作 | 位与参数 |
| --- | --- |
| 两个 Hadamard | a0、a1 |
| 两个受控 RX | a0控制s0，theta0；a1控制s1，theta1；控制值1 |
| 第一个 RZZ | s0、s1，theta2 |
| 两个 RX | s0，theta3；s1，theta4 |
| 第二个 RZZ | s0、s1，theta5 |

论文门是

$$
R_X(\theta)=e^{-i\theta X},\qquad R_{ZZ}(\theta)=e^{-i\theta Z\otimes Z},
$$

$$
CR_X(\theta)=|0\rangle\langle0|\otimes I+|1\rangle\langle1|\otimes R_X(\theta).
$$

因为 $X^2=I$，$R_X(\theta)=\cos\theta I-i\sin\theta X$。库的 RX 则将角度除以2，所以要传入 `2*theta`；受控 RX 复用同一 RX 定义。RZZ 的 `half=True` 也显式保留，配合两倍角度。每一种门都已独立验证，不能把这个接口约定推广到未经核对的其他门。Hamiltonian 中 ZZ 的1/4系数属于动力学模型，不乘进这里的门定义。

`purification_state(theta)` 返回线路态 $|\Psi(\theta)\rangle$，shape `(16,)`、complex128；仍是 JAX 数组。核心线路和约化函数没有把参数转换为 Python float 或 NumPy，也没有读取 Exact 数据，因此保留了下一阶段求导所需的计算路径，但求导正确性仍需 Phase 4 实际验证。

## purification 为什么能表示混态

在熟悉的量子力学中，整体纠缠态可以是纯态，而其中一个子系统由混合密度矩阵描述。这里把两个辅助位视为不观测的子系统；只保留系统位。所谓 purification，是用较大空间中的纯态表示较小系统的混态，并不是把密度矩阵的概率人为归一化。

将振幅写为

$$
|\Psi\rangle=\sum_{a,s} A_{as}|a\rangle_A|s\rangle_S.
$$

完整密度矩阵是 $\sum_{a,b,s,t}A_{as}A_{bt}^*|a\rangle\langle b|\otimes|s\rangle\langle t|$。利用 $\operatorname{Tr}_A(|a\rangle\langle b|)=\delta_{ab}$，得到严格等式

$$
(\rho_S)_{st}=\sum_a A_{as}A_{at}^*.
$$

`reduce_system_density(state)` 先用 `jnp.reshape(state,(4,4))` 得到行索引是辅助位、列索引是系统位的 A，再用

```python
rho = jnp.einsum("as,at->st", amplitudes, jnp.conj(amplitudes))
```

执行上式。相同字母 a 表示求和，箭头右侧 st 是保留下来的两个系统矩阵下标。`conj` 是复共轭。它等价于 `A.T @ A.conj()`；`A @ A.conj().T` 会得到辅助位的密度矩阵，而 `A.conj().T @ A` 的共轭位置不符合上式。复振幅测试用例能够区分这些看似相近的写法。

`system_density(theta)` 将生成纯态与部分迹连接起来，输出 shape `(4,4)`、complex128。函数不会重新归一化：$\operatorname{Tr}\rho_S=\langle\Psi|\Psi\rangle$。验证还将输入向量乘2，确认约化矩阵随之乘4，而不是被强制改成迹1。

## 初态和可解析极限

零参数时全部旋转为单位算符，但两个 H 仍然存在，所以

$$
|\Psi(0)\rangle=|++\rangle_A|00\rangle_S,
\qquad \rho_S(0)=|00\rangle\langle00|.
$$

完整向量仅在下标0、4、8、12处有振幅1/2。这也是对基底顺序的检查，不能要求整体态仍为 $|0000\rangle$。

当 $\theta_0=\theta_1=\pi/2$ 时，两个受控门的活跃分支为 $-iX$；每一对辅助/系统位形成等权纠缠。取部分迹后 $\rho_S=I_4/4$，后续任意系统幺正门也不改变它。这一非纯态极限已在17组线路样本中验证。

## 验证结果、容差与限制

所有比较均定义为最大绝对矩阵元素差；厄米性用 Frobenius 范数。容差为1e-12。17组参数包括零点、逐参数非零、两个通用例子、8个种子2019的随机例子；部分迹另有6个直积/复振幅/纠缠/随机例子，随机态种子2026。参考路径直接构造16×16 NumPy/SciPy门矩阵，不调用 TensorCircuit 门；部分迹参考显式求和完整密度矩阵的辅助对角块，并与 QuTiP ptrace 比较。

实际最大完整态误差约1.570e-16，系统密度误差约2.225e-16；迹误差6.661e-16，厄米误差0。最小计算特征值约−1.621e-17，位于允许的−1e-12容差以内；没有裁剪该数值，也不将有限精度检查声称为严格数学证明。数学上，上面的约化公式保证厄米性；对任意 x，$x^\dagger\rho_S x=\sum_a|\sum_s x_s^*A_{as}|^2\ge0$，保证半正定性。

位序的单门基矢检查、系统Z0对应wire2、Z1对应wire3、前向JIT以及float64/complex128精度均通过。全部回归测试16项通过；没有新增依赖。样本NPZ和元数据位于 `results/ansatz_20261014_152519_728727/`，运行命令见 README。

Phase 3 只证明给定参数时，线路与约化态在这些独立检查中正确。未计算密度矩阵 Jacobian、初点的切空间/M秩，也没有推进任何参数时间轨迹。下一阶段需经用户验收后单独执行。
