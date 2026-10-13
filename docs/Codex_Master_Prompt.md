# Codex Master Prompt — Yuan et al. (2019) Figure 5

请作为我的科研编程助手，分阶段复现 Yuan et al. (2019), *Theory of variational quantum simulation*, Quantum 3, 191 的 Figure 5。以下内容是用户对本项目的明确要求。首次执行只完成 Phase 0 的检查与规划，随后停下等待我验收，不能直接进入完整实现。

## 1. 已确认需求与边界

- 采用分阶段开发，每个阶段完成、验证、解释结果后停止；只有我明确批准才能进入下一阶段。当前只做仓库创建/关联准备与规划。
- 函数名、代码注释、Docstring 和 README 使用英文；独立中文学习文档解释公式、物理意义与 Python 操作。
- 基础交付：Figure 5 图片、Exact 与 VQS 数值数据、完整代码及运行必需的文档、测试、依赖记录。不额外制作参数曲线、诊断图或长篇复现报告；必要诊断保存在同一数值文件或简洁运行记录中。
- 环境：项目内 `.venv`，使用 venv + pip；不使用 uv/Conda，不改变系统 Python 或其他项目环境。安装依赖须获得针对环境阶段的明确授权，已有授权不重复询问。
- Git：独立开发分支，阶段通过必要检查后进行本地 commit；未经我明确许可不 push、不 merge，也不通过 GitHub 文件 API 绕过这一限制。
- 不为这个小项目搭建通用框架、复杂类层次、无用目录或自动化服务。

实际工作区：`/Users/ornfr/Desktop/ai小垃圾/MSVQD`

唯一可写项目目录：`/Users/ornfr/Desktop/ai小垃圾/MSVQD/reproductions/yuan2019_fig5/`

只读文献目录：`/Users/ornfr/Desktop/bookandpaper/`。它实际位于 MSVQD 外部，不要假设是 MSVQD 内的子目录。不得创建、修改、移动、重命名或删除文献，也不得向它写入缓存、标注和解析结果；不能复制或通过符号链接将它纳入 Git。

先阅读工作区 README.md、AGENTS.md、PROGRESS.md 及项目内已有文件。保留工作区内其他研究任务的状态，不改动其文件。上述用户指定的英文代码/中文学习文档和独立项目结构优先于父级规范的通用语言与目录建议。项目进度记录放在本子项目自己的 PROGRESS.md。

GitHub：`https://github.com/ORnFr/yuan2019-fig5`，Private。Git 根只能是本子项目，不能是整个 MSVQD。先核实远程和本地实际状态；网页仓库已有 main 分支及初始 README 提交，不能默认远程为空。如果 Git 尚未认证，记录阻碍并指导完成认证；在正确拉取远程基线前不要创建另一套独立提交历史。不得强推、重置用户变更、删除 `.git` 或自行合并不相关历史。

## 2. 依据、来源与假设

主要来源：

- [期刊正式论文](https://quantum-journal.org/papers/q-2019-10-07-191/)
- [期刊 PDF](https://quantum-journal.org/papers/q-2019-10-07-191/pdf/)
- [arXiv:1812.08767v4](https://arxiv.org/abs/1812.08767v4)

在正式编码前核对本地论文版本；以 Figure 4、Figure 5、Eq. (28)、Eq. (29)、Eq. (62) 为主。v4 中 Eq. (28)/(29) 在印刷第 11 页，Figure 4 与模型在第 20 页，Figure 5 在第 21 页；PDF 页序以实际文件为准。必须目视阅读线路，不能仅依赖 PDF 文本提取。

文档明确区分：论文明确条件；用户指定方案；本项目必要的数值选择；待确认问题。Forward Euler 是本项目按用户要求采用的基线，不声称论文已完整交代其积分实现。SVD 阈值、正则化、QuTiP 容差、数据格式及软件版本也属于本项目选择。假设使用 ħ=1，按下述哈密顿量系数定义无量纲时间。

## 3. 技术路线与精度

- Exact：QuTiP `mesolve` 求解 Lindblad 主方程，作为独立的高精度数值参考。
- VQS：TensorCircuit 构造四量子比特纯态线路，JAX 后端计算部分迹和参数导数。
- NumPy/SciPy：边界处的数据保存、独立矩阵参考与必要线性代数。
- Matplotlib 绘图，pytest 验证科学计算模块。
- 在任何数组、线路或 JIT 创建前启用 JAX 64 位，设置 TensorCircuit complex128，核实实际输出是 float64/complex128。
- 在 Mac 上优先用 CPU；不得假设 GPU/Metal 已可用或为本项目安装相关插件。
- 环境阶段先检查 Python、芯片架构和实际兼容版本。固定已验证的依赖，不凭记忆指定“最新可兼容版本”。保存 Python 和直接依赖版本；`pip check` 与小型导入/求导检查通过才进入下一阶段。
- 不在可微函数内转成 NumPy、QuTiP 对象或 Python float，不切断 JAX 计算图。不用有限差分取代正式自动微分。

## 4. Exact Lindblad 模型

系统基底顺序：`|s0 s1>` = `|00>, |01>, |10>, |11>`。定义

\[
X_0=X\otimes I,\quad X_1=I\otimes X,\quad Z_0=Z\otimes I,
\qquad H=X_0+X_1+\tfrac14 Z_0Z_1.
\]

\[
\rho(0)=|00\rangle\langle00|,\qquad
\sigma^-=|0\rangle\langle1|=\begin{pmatrix}0&1\\0&0\end{pmatrix},
\quad \sigma^+=(\sigma^-)^\dagger.
\]

Eq. (62) 的生成元：

\[
\mathcal L(\rho)=-i[H,\rho]+
\sum_{j=0}^{1}\left(\sigma_j^-\rho\sigma_j^+
-\tfrac12\{\sigma_j^+\sigma_j^-,\rho\}\right).
\]

QuTiP collapse operators 是 `c0 = sigma_minus ⊗ I`、`c1 = I ⊗ sigma_minus`，即 gamma=1。式中的 1/2 是反对易项的标准系数，不意味着 collapse operator 再乘 1/sqrt(2)。建议用 `basis(2,0) * basis(2,1).dag()` 明确构造，避免仅凭 `sigmam`/`sigmap` 名称判定计算基底中的升降方向；如使用库函数，先断言矩阵及对基矢的作用。

输出时间网格 `t_n=n*0.01, n=0,...,1000`，共 1001 个点，包含 t=0 与 t=10。QuTiP 内部允许自适应积分；0.01 是保存网格，不是要求 Exact 用 Euler。明确记录求解容差，用更严格容差抽查结果，确保参考误差明显小于 1e-2 的比较目标。

计算 `z0_exact(t)=Re Tr(Z0 rho_exact(t))`。Exact 模块独立运行；与手写生成元或 QuTiP Liouvillian 的独立小型对照验证符号、耗散率和张量顺序。

## 5. Figure 4 四量子比特 purification ansatz

六个实参数采用论文编号 `theta=(theta0,...,theta5)`，Python shape `(6,)`。

固定一个清楚的实现顺序，例如 TensorCircuit wires `(a0,a1,s0,s1)=(0,1,2,3)`，并在代码与文档中声明；不同框架之间显式映射系统量子比特 0，不能把 wire 0 的辅助位当成 Z0 的系统位。

从 `|0000>` 开始，按时间从左向右执行：

1. `H(a0)` 与 `H(a1)`。
2. `controlled-RX(theta0)`，控制 a0，目标 s0，控制值为 1。
3. `controlled-RX(theta1)`，控制 a1，目标 s1，控制值为 1。
4. 系统上 `RZZ(theta2)`。
5. 系统上 `RX(theta3)` 作用于 s0，`RX(theta4)` 作用于 s1。
6. 系统上 `RZZ(theta5)`。

旋转定义是

\[
R_X(\theta)=e^{-i\theta X},\quad
R_{ZZ}(\theta)=e^{-i\theta Z\otimes Z},\quad
CR_X(\theta)=|0\rangle\langle0|\otimes I+
|1\rangle\langle1|\otimes e^{-i\theta X}.
\]

前两个门是 controlled-RX，不能改为 CNOT 或无控制的 RX。按 Figure 4 实现；H 的 1/4 系数用于动力学生成元，不额外塞进 RZZ 的定义或参数更新。

TensorCircuit 的 RX 通常使用 `exp(-i*angle*X/2)`，因此论文 theta 对应库 angle=2*theta；CRX 与 RZZ 也必须逐个核对当前版本的矩阵定义及 half 选项。不要假定所有门具有相同接口。用独立稠密矩阵指数在几个非零角度验证单门、受控门和整条线路，包括控制=0时目标不变化。

生成 `psi(theta)` shape `(16,)`，并计算

\[
\rho_S(\theta)=\operatorname{Tr}_{a0,a1}
\bigl[|\Psi(\theta)\rangle\langle\Psi(\theta)|\bigr].
\]

若已验证基底顺序 `(a0,a1,s0,s1)`，可把 psi reshape 为 `A[a,s]` shape `(4,4)`，然后 `rho[s,t]=sum_a A[a,s]*conj(A[a,t])`。不得不核对顺序便使用 reshape 或 `A @ A.conj().T`。用直积态、纠缠态及独立的完整 16×16 密度矩阵部分迹比较验证。

全零参数时系统为 `|00><00|`，但辅助位因 Hadamard 已处于 `|++>`。不要错误要求整个四位态仍是 `|0000>`。

## 6. 自动微分与 McLachlan 方程

用实参数对复密度矩阵求导，得到

\[
D_i=\partial\rho_S/\partial\theta_i,
\qquad D\ \text{shape}\ (6,4,4).
\]

可分别对 rho 的实部与虚部做 `jax.jacfwd`，再合并，并显式将参数轴从末尾移至首位；或使用已验证支持实输入/复输出的 JVP/jacfwd 路径。rho(theta) 含共轭，不是复参数的全纯映射，不能用 `holomorphic=True` 掩盖类型问题。

按 Eq. (28)/(29) 实现

\[
\sum_j M_{ij}\dot\theta_j=V_i,\qquad
M_{ij}=\operatorname{Re}\operatorname{Tr}(D_i^\dagger D_j),\quad
V_i=\operatorname{Re}\operatorname{Tr}[D_i^\dagger\mathcal L(\rho_S)].
\]

保留共轭转置的 Hilbert–Schmidt 内积；在实参数、Hermitian 导数下可化简，但必须先验证。M shape `(6,6)`，V shape `(6,)`，theta_dot shape `(6,)`。验证 M、V 的虚部仅为数值舍入量后取实部，不能用取实部隐藏错误。

必须使用约化系统密度矩阵的切向量，不能替换成四位 purification 纯态的 Fubini–Study/纯态度量。V 的耗散和 Hamiltonian 项必须全部来自当前 VQS rho。不得用 Exact 的状态、观测量或 Figure 5 数据拟合 theta。

## 7. 秩亏、稳定性与时间演化

theta(0)=六个零；首先计算并报告该点的 Jacobian、M 的秩、奇异值和 V。纯态起点可能具有冗余或一阶不可见方向，不假设 M 可逆，也不声称伪逆能修复 ansatz 的所有切空间限制。

基线使用明确阈值的 SVD 最小范数最小二乘/截断伪逆，而非 `inv(M) @ V`。先展示阈值方案及理由，例如按 dtype、维数和最大奇异值定义 cutoff；固定配置、记录 rank、保留谱的条件数、线性残差 `||M theta_dot - V||`、切空间残差 `||sum_i D_i theta_dot_i - L(rho)||_F`、参数步长及 NaN/Inf。秩亏矩阵的完整条件数应报告为无穷或秩亏，不能把保留谱的条件数误称完整条件数。

对阈值作小规模敏感性检查；若需 Tikhonov、非零扰动初态、改变参数化、积分器或首步特殊处理，先给出诊断证据并提出方案，停下等待我批准。这些改变会影响基线，不能静默采用，尤其不能改成非零初参来强行贴合论文图。

首先采用用户指定的 Forward Euler：

\[
\theta_{n+1}=\theta_n+0.01\dot\theta_n,
\qquad n=0,...,999.
\]

每步在更新前的 theta_n 计算 rho、D、M、V，更新后重建合法密度矩阵；不要用 rho 的 Euler 更新替代参数演化。保存 1001 个时间点及对应参数，避免观测量与时间错一格。为诊断可以短时比较更小步长；最终基线仍为 dt=0.01，不能用更小步长结果冒充。

## 8. 科学验证与代码解释

每个核心函数有英文 Docstring，写明物理意义、LaTeX 公式、输入输出、shape、dtype、基底/单位与来源方程。关键张量操作、求导、线性求解与时间更新有解释性英文注释。中文学习文档逐模块把公式与函数对应起来，适合有量子力学基础、Python 经验较少的研究生。保持函数短而可读，避免过度封装；优化/JIT 在正确性得到验证后进行。

必要验证包括：

- 模型矩阵、collapse operator 对基矢的作用；H=0 单量子比特阻尼的解析极限，核对人口指数衰减率。
- 线路门矩阵、门顺序、控制方向、零参数系统初态；系统与辅助位映射。
- 部分迹与独立参考一致；rho 的 trace=1、Hermiticity、最小特征值容差。
- 在零参数和固定种子的非零参数上，对多种有限差分步长比较 AD；有限差分仅用于验证，记录绝对与相对误差。
- D_i 的 Hermiticity 与 trace≈0；M 对称半正定，M、V 实数性质。
- 奇异点求解可解释、残差合理、无 NaN/Inf；短时间 Euler 能运行。
- 全过程物理合法性、文件时间点数量与参数/观测量对齐。
- Exact 与 VQS 可分别运行；VQS 不读取 Exact 数据，绘图模块才读取两者比较。

容差必须写明并与 64 位舍入及参考精度匹配，不能把数值测试写成只复述同一实现。未执行的测试标为未执行，不编造通过结果。

## 9. 最终输出与整洁目录

按需要逐阶段创建，规划结构如下，不要现在预建全部空目录：

```text
yuan2019_fig5/
  README.md
  AGENTS.md
  PROGRESS.md
  .gitignore
  requirements.txt
  docs/
    Codex_Master_Prompt.md
    github_setup_zh.md
    learning_zh.md
  src/yuan2019_fig5/
    __init__.py
    model.py
    exact.py
    ansatz.py
    derivatives.py
    mclachlan.py
    vqs.py
    plot.py
  tests/
  results/
    data/exact.npz
    data/vqs.npz
    data/run_metadata.json
    figures/figure5.png
  work/                 # ignored temporary files, only when needed
  .venv/                # ignored, only in approved environment phase
```

这是建议划分，可用更少文件合并紧密相关的小模块，但保持 Exact、VQS 与绘图可独立运行。不要同时维护两套配置或重复入口。输出路径相对于项目/脚本定位，避免依赖当前终端目录。README 给出可实际执行的创建环境、安装、测试、Exact、VQS、绘图命令。

Exact 数据至少含 t、rho `(1001,4,4)`、z0；VQS 至少含 t、theta `(1001,6)`、rho、z0，必要诊断也放入 vqs.npz。元数据记录模型、精度、依赖、Euler dt、Exact 容差、SVD 策略及版本来源；这些是重现必需记录，不是额外长报告。

Figure 5：主图为 Exact 线与 VQS 点的 Z0 随时间变化；小插图显示带符号差值 `error=z0_vqs-z0_exact`。如抽稀点标记，仅影响显示，完整数据与最大误差仍使用所有 1001 个点。不新增独立误差图、参数图或稳定性图。

报告 `max_t |error(t)|`，将论文小于 1e-2 的偏差作为对照目标；未达到就忠实报告，不调物理参数、不隐藏异常、不用 Exact 拟合、不声称视觉相似就是完整复现。

Git 只跟踪本子项目的代码、文档、小规模复现数据和最终图。忽略 `.venv/`、缓存、临时日志、`work/`、`.DS_Store`、密钥、外来论文原件和 `bookandpaper/`。不能因加了 gitignore 就认为已跟踪文件被移除；提交前检查 staged 文件清单，显式添加本阶段路径，避免在父目录 `git add .`。保留最终 NPZ/JSON/PNG，不将 results 整体忽略。

## 10. 分阶段验收

| 阶段 | 工作与验收证据 | 停止点 |
| --- | --- | --- |
| Phase 0 | 读取规范、论文与已有文件；核实仓库/路径/环境；展示模型约定、目录、依赖候选、奇异性风险和阶段计划 | 只规划，等待批准 Phase 1 |
| Phase 1 | 经授权建立 .venv、安装并固定兼容依赖；通过 pip check、64位与基础求导检查 | 报告结果，本地阶段 commit，停止 |
| Phase 2 | Exact 模型、生成元与 QuTiP 演化；解析阻尼和物理约束验证；保存 Exact 数据 | 解释结果，commit，停止 |
| Phase 3 | Figure 4 线路、门参数因子、purification 与部分迹；独立矩阵和初态验证 | 解释结果，commit，停止 |
| Phase 4 | AD Jacobian；有限差分对照；初点及非零点的切向量性质 | 解释结果，commit，停止 |
| Phase 5 | M、V、SVD 策略与 Euler；短时验证后运行 t≤10；记录诊断与 VQS 数据 | 解释结果，commit，停止 |
| Phase 6 | Figure 5 主图/误差插图、全点误差、必要验收；完善运行说明与中文学习文档 | 本地 commit，提交验收；仍不 push/merge |

每阶段只执行该阶段已授权工作，先处理它内部的常规修复，再交付：修改内容、可运行命令、实际验证、限制/假设、下一阶段待批准范围。更新本子项目 PROGRESS.md。Git 认证或身份缺失时明确记录 commit 尚未完成，不自行虚构身份。基础成果之外的扩展必须先征得我的同意。

现在请先执行 Phase 0：核实当前状态并向我展示具体计划，然后停止，等待我验收。
