# Phase 1：环境与基础求导检查

后续已完成的 Exact 演化说明见 [notes/exact_lindblad_zh.md](../notes/exact_lindblad_zh.md)，原图核对见 [papers/figure5_exact_source_zh.md](../papers/figure5_exact_source_zh.md)。Figure 4线路与部分迹见 [notes/ansatz_purification_zh.md](../notes/ansatz_purification_zh.md)，密度矩阵Jacobian见 [notes/density_derivatives_zh.md](../notes/density_derivatives_zh.md)，McLachlan/SVD与参数演化见 [notes/mclachlan_vqs_zh.md](../notes/mclachlan_vqs_zh.md)，最终Figure 5与带符号误差见 [notes/figure5_comparison_zh.md](../notes/figure5_comparison_zh.md)。下文保留Phase 1学习记录。

日期：2026-10-13。这里只验证计算工具，没有开始论文 Figure 5 的时间演化。英文运行命令见项目 README，版本选择与实际数值证据见 `docs/environment.md` 和 `results/phase1_environment_20261013_212040_737785.json`。

## 独立环境与数据精度

`.venv` 相当于本项目独立的 Python 依赖目录。使用 `.venv/bin/python` 时，导入的是本项目安装的库，不需要修改系统 Python。当前使用已经存在的 arm64 Python 3.12.14 创建环境；基础解释器和标准库仍由原位置提供，因此不能把 `.venv` 当作可以随意搬动的完整 Python 安装包。

`requirements.txt` 的 `包名==版本` 固定了实际验证过的直接依赖和间接依赖。`pip check` 检查软件包声明的版本约束是否互相满足，但它不能证明物理计算正确，所以还要用有解析答案的小线路进行检查。

`float64` 表示用 64 位保存一个实数，`complex128` 表示实部和虚部分别采用 float64。JAX 默认没有启用 64 位计算，仅在变量名中写“64”或请求 dtype 不足以证明实际精度。本项目在创建可计算数组、线路和 JIT 前启用 `jax_enable_x64`，并设置 TensorCircuit `complex128`。检查还要求在 1 附近加上 `2^-40` 后差值保留下来；这种增量在 float32 中会消失。验证观察到实数和复数虚部都保留了该增量。

脚本将 CPU 设为唯一计算后端；缓存保存在项目内的 `work/`。QuTiP 的导入默认行为需要单独处理：`import_project_qutip()` 仅在导入期间将其默认缓存位置引导到项目内，随后恢复路径函数。后续脚本不能先直接导入 QuTiP，再补设缓存路径。

## 用量子力学解析答案检查线路

单比特希尔伯特空间是 $\mathbb C^2$，基底顺序为 $|0\rangle,|1\rangle$，初态为 $|0\rangle=(1,0)^T$。角度 $a$ 以弧度计，是无量纲参数；这里不设置物理哈密顿量、时间、边界条件或耗散模型。

环境测试采用 TensorCircuit 的门约定：

$$
R_x(a)=\exp(-iaX/2),\qquad
X=\begin{pmatrix}0&1\\1&0\end{pmatrix}.
$$

因为 $X^2=I$，指数的偶次和奇次项分别给出余弦和正弦，所以

$$
R_x(a)=\cos(a/2)I-i\sin(a/2)X,
\qquad
|\psi(a)\rangle=\begin{pmatrix}\cos(a/2)\\-i\sin(a/2)\end{pmatrix}.
$$

对应 `rx_state(angle)`。输出数组的 shape 为 `(2,)`，两个元素就是上述两个复振幅。独立参考用 SciPy 计算矩阵指数后作用于 $|0\rangle$；这条参考路径没有调用 TensorCircuit 门构造。

对 $Z=\operatorname{diag}(1,-1)$，归一化和观测量为

$$
\langle\psi|\psi\rangle=1,\qquad
\langle Z\rangle=\cos^2(a/2)-\sin^2(a/2)=\cos a.
$$

对应 `z_expectation(angle)`。**这个测试中的库角度 $a$ 不是论文的参数 $\theta$。** 论文规划采用 $\exp(-i\theta X)$，后续线路阶段需要验证 $a=2\theta$；本阶段没有实现论文线路。

双比特检查在 $\mathbb C^2\otimes\mathbb C^2$ 上从 $|00\rangle$ 出发，按顺序施加第 0 位 Hadamard 和控制位为 0、目标位为 1 的 CNOT，得到

$$
\frac{|00\rangle+|11\rangle}{\sqrt2}.
$$

`check_bell_circuit()` 比较 shape `(4,)` 的输出与解析向量 $(1,0,0,1)^T/\sqrt2$，基底顺序为 $|00\rangle,|01\rangle,|10\rangle,|11\rangle$。这只是小型纠缠线路检查，没有实现辅助位、purification 或部分迹。

## 自动微分为什么需要独立检查

解析导数为

$$
\frac{d\langle Z\rangle}{da}=-\sin a,\qquad
\frac{d|\psi\rangle}{da}=\begin{pmatrix}-\tfrac12\sin(a/2)\\-\tfrac i2\cos(a/2)\end{pmatrix}.
$$

`jax.grad` 用于实标量观测量的梯度；`jax.jacfwd` 用于实参数到复态向量的导数。这里不使用 `holomorphic=True`：参数是实数，不需要假定函数对复参数全纯。可微函数内部始终保留 JAX 数组，转换为 NumPy 或 Python float 只在求导完成后的比较与记录中发生。

`jax.jit` 将计算编译后执行，`jax.vmap` 将同一函数应用到一组输入。测试要求编译和批量计算后仍保留正确数值。TensorCircuit 自己的 JAX 后端求导接口也与解析结果比较。

验证还使用中心差分

$$
\frac{\psi(a+h)-\psi(a-h)}{2h}
$$

作为独立检查，**没有用有限差分替代后续正式自动微分**。固定输入为 $a=0,0.37,-0.83,1.2,\pi$；无随机数。误差定义为这些输入上各数组元素的最大绝对差，不是相对误差。解析/矩阵比较的容差为 $10^{-12}$；有限差分采用三个步长及分别记录的容差，因为它本身含有离散化误差。

实际最大态误差和梯度误差均约 $1.11\times10^{-16}$；复态导数在选定输入上的最大误差为 0（仅说明这些浮点比较相同，不代表任意输入精确无误差）。有限差分误差在 $h=10^{-4},10^{-5},10^{-6}$ 时约为 $2.08\times10^{-10},4.91\times10^{-12},8.47\times10^{-11}$，全部低于各自容差。仅凭这三个点不推断完整收敛区间或误差来源。

pytest 的 4 项测试已实际通过。结果证明当前小型 CPU 线路与 AD 路径可用；主方程积分、论文完整线路、部分迹和变分方程仍需各阶段的科学验证。Phase 1 完成后停止，等待验收与 Phase 2 的单独授权。
