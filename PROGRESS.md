# Figure 5 reproduction progress

日期：2026-10-14（Asia/Shanghai）。用户以“A”验收Phase 5并批准Phase 6；Figure 5主图/带符号误差插图与全点比较已完成，1001点最大绝对误差0.00755849<0.01。停下等待最终验收，不自动扩展研究或上传。

## 完成与验证

- 读取父工作区规范，保留其他研究项目；所有改动仅在本子项目。
- GitHub 私有仓库：ORnFr/yuan2019-fig5；默认分支 main。
- Git 根确认是本子项目；origin=https://github.com/ORnFr/yuan2019-fig5.git。
- 用户完成 Git 认证后，实际 fetch 成功；远程初始提交为 e7927b483d7d83b0e67a86df16ab009e89eadb97。
- 本地 main 与该提交一致并跟踪 origin/main，保留本地规划文件，创建 phase0/planning 分支。
- 经用户确认，仅在本仓库设置作者 ORnFr、307864480+ORnFr@users.noreply.github.com；未改变全局身份。
- 完成 Master Prompt、关联指南、英文 README、项目规范、.gitignore；本阶段文档随规划 commit 保存，提交编号通过 git log 查看。
- 已目视核对 Figure 4，核对模型与旋转/升降算符约定；不把规划检查称为数值复现。
- 提交前检查允许的文件清单、空白错误、Markdown围栏与文档副本一致性。

## Phase 1 完成与验证

- 开始前实际确认 Git 根、干净工作区、规划分支 `phase0/planning`、规划提交 `07ea9d7`、origin 和仓库内作者身份；没有重新初始化、重置或覆盖提交。
- 创建本地 `phase1/environment` 分支，保留 main 与规划分支。阶段 commit 编号以 `git log -1 --oneline phase1/environment` 查询，避免在其自身内容中记录自引用哈希。
- 本机 macOS 26.6.2、arm64；默认 Python 3.14.6。使用已经存在的 Codex-bundled arm64 Python 3.12.14 创建独立 `.venv`，未安装基础解释器，未修改其文件。
- 查阅官方文档与精确 PyPI 元数据；以 wheel-only 安装 QuTiP 5.2.3、TensorCircuit 0.12.0、JAX/jaxlib 0.4.38、NumPy 1.26.4、SciPy 1.15.3、Matplotlib 3.11.2、pytest 9.1.1。`requirements.txt` 固定全部 27 个已安装分发包（含 pip 25.0.1）；离线 dry-run 确认所有 pins 已满足。
- `pip check` 实际返回 `No broken requirements found.`；所有 pins 与实际版本一致。依赖选择、缓存诊断和限制见 [docs/environment.md](docs/environment.md)，安装 wheel 来源与 SHA-256 见 [results/phase1_dependencies.json](results/phase1_dependencies.json)。
- 环境脚本实际验证 CPU、JAX x64、TensorCircuit JAX 后端及 float64/complex128；在 1 附近的 `2^-40` 实数/虚部增量被保留。QuTiP 导入缓存也限定在本项目 `work/`，不改变 HOME。
- 小型 RX、双比特 Bell 线路、独立矩阵参考、解析求导、复态 `jacfwd`、有限差分、JIT/vmap、TensorCircuit 后端 AD、QuTiP 矩阵与 Matplotlib Agg 检查通过。态/梯度最大绝对误差约 `1.11e-16`，容差 `1e-12`；有限差分误差与分步长容差保存在 [最终环境报告](results/phase1_environment_20261013_212040_737785.json)。不把环境检查称为论文复现。
- 从 `/private/tmp` 启动 pytest，最终 `4 passed in 1.84s`；缓存和日志仍在子项目。英文运行命令见 README，独立中文解释见 [docs/learning_zh.md](docs/learning_zh.md)。

## Phase 2：Exact 曲线

- 从干净的 Phase 1 提交 `d9d208d` 创建本地 `phase2/exact`；保留已有分支与历史。当前阶段提交用 `git log -1 --oneline phase2/exact` 查询。
- 实际阅读、提取并目视核对只读论文 PDF/印刷第 20–21 页的 §6.3、Eq. (62)、Figure 5。来源、文件哈希、原文条件、比较范围见 [论文记录](papers/figure5_exact_source_zh.md)；没有复制或修改论文。
- 实现固定两位 Lindblad 模型、显式计算基底 lowering operators 和独立 Exact 入口。QuTiP `mesolve` 使用 DOP853、atol=1e-12、rtol=1e-10、关闭输出归一化；保存 `t_n=0.01*n` 的全部 1001 个点，无拟合、平滑、时间平移、特征值裁剪或变分数据输入。
- 完成 Hamiltonian/collapse 矩阵、独立手写生成元、解析单比特阻尼、全点独立矩阵指数、更严格 ODE 容差以及 trace/Hermiticity/positivity 检查。独立参考的最大 rho 元素误差 `2.844e-15`、Z0 误差 `2.276e-15`（容差1e-9）；迹误差 `1.110e-15`、厄米误差 `4.170e-16`、最小特征值0（物理容差1e-10）。
- 从项目外 `/private/tmp` 实际运行合并测试：`7 passed in 1.98s`；`pip check` 仍通过，没有新增或改变依赖。缓存路径逻辑抽到共享 runtime；独立 Exact 实际运行不导入 JAX/TensorCircuit，原环境检查也回归通过。
- 数据、Exact-only 图片与验证/软件/代码哈希元数据位于 [results/exact_20261013_220047_446893](results/exact_20261013_220047_446893/)；NPZ 已读回逐元素核对，PNG 已目视检查。运行命令见 README，数学与代码说明见 [notes/exact_lindblad_zh.md](notes/exact_lindblad_zh.md)。
- 结果：Z0(0)=1；输出网格谷值约−0.120574982，位于t=1.57；Z0(10)≈0.122015654。
- 原图蓝线提取940个矢量顶点，首点(t=0.01,Z0=1)。直接按图坐标最大差0.0110552；一个“1000样本覆盖0…10却显示于0.01…10”的采样解释将差降到1.790e-6。该解释是未确认的绘图假设，不是作者原始数据或源代码；交付数据未应用它。详细证据在论文记录和run_metadata.json，不能声称发表图已逐点完全重合。

## Phase 3：Figure 4 线路与部分迹

- 从干净且已验收的 Exact 提交 `41654c3` 创建本地 `phase3/ansatz`；之前的分支/提交保留。本阶段提交编号用 `git log -1 --oneline phase3/ansatz` 查询。
- 重新目视核对论文 PDF/印刷第20页 Figure 4，核对已安装 TensorCircuit 0.12.0 的 RX、controlled 和 RZZ 源码；[来源及逐门约定](papers/figure4_ansatz_source_zh.md)区分原文定义和软件位序选择。
- 新增 `ansatz.py`：位序(a0,a1,s0,s1)=(0,1,2,3)，六个实参数，严格门顺序，控制值1。RX/CRX调用两倍角度；RZZ调用两倍角度并显式half=True，不插入Hamiltonian的1/4。整体态/系统态均为complex128，保留JAX路径。
- 部分迹用 `rho[s,t]=sum_a A[a,s]*conj(A[a,t])`；不归一化、不修复状态。零参数整体态是|++>_A|00>_S，系统态是|00><00|；最大纠缠例子的系统态I4/4也通过验证。
- 17组参数（零、单参数、通用、种子2019随机）与独立16×16矩阵计算比较：最大态元素差1.570e-16、最大rho元素差2.225e-16，容差1e-12。门矩阵全列/控制0、完整密度矩阵块求和、6个部分迹例子（随机态种子2026）和QuTiP ptrace均通过。
- 迹误差6.661e-16、厄米Frobenius误差0、最小计算特征值−1.621e-17（容差−1e-12）。位有效次序、Z0对应wire2、Z1对应wire3、前向JIT及CPU/64位精度通过；没有计算ansatz导数。
- 最终17组验证参数、纯态及系统密度矩阵保存在 [results/ansatz_20261014_152519_728727](results/ansatz_20261014_152519_728727/)，附版本、容差、实际误差和代码/数据哈希；NPZ已读回逐元素检查。不是VQS时间序列，没有使用Exact数据拟合参数。
- 从项目外实际运行合并测试：`16 passed in 2.63s`；`pip check`通过；未新增依赖。Exact核心代码、已验收结果和固定依赖均无改动。运行命令见README，[中文公式与代码说明](notes/ansatz_purification_zh.md)已完成。
- 已把用户要求的“完成反馈提供2–3个简单下一步选项”写入本项目AGENTS.md，供后续阶段沿用。

## Phase 4：系统密度矩阵Jacobian

- 从干净的ansatz提交 `3573a98` 创建本地 `phase4/derivatives`，保留先前分支。阶段提交编号用 `git log -1 --oneline phase4/derivatives` 查询。
- 新增 `derivatives.py`：分别对rho实部、虚部 `jax.jacfwd`，合并并将参数轴从末尾搬到首位，输出complex128 `(6,4,4)`。参数是实弧度，没有holomorphic=True、NumPy转换、FD替代AD或导数修复。
- 实际提取并目视核对只读论文PDF/印刷第11页Eq.(26)、(28)/(29)。相关段落i/j下标不一致不作为不同参数导数相等的依据；严格逐参数验证D_i†=D_i。详细[来源与方法记录](papers/phase4_derivatives_source_zh.md)和[中文推导](notes/density_derivatives_zh.md)已完成。
- 17组零/非零参数（随机种子2019）的AD与独立NumPy/SciPy门乘积法则、完整密度导数块求和比较：最大元素误差5.551e-16；零点独立commutator解析误差4.441e-16，容差1e-12。
- 初点D2=D5=0、D3=2D0、D4=2D1，六个Frobenius范数约(0.7071,0.7071,0,1.4142,1.4142,0)，只有两个独立第一阶系统密度方向。保留完整Jacobian和零参数初态，不通过扰动或截断修改它；没有计算M/V/谱或求解。
- 四个参数FD步长1e-3/1e-4/1e-5/1e-6的最大绝对差分别6.667e-7/6.667e-9/6.667e-11/1.474e-10，满足逐步长绝对与相对容差。相对误差以独立导数Frobenius范数为分母；12个范数≤1e-12方向标记为未定义，未当作零相对误差。
- 导数迹最大误差2.238e-16、厄米Frobenius误差0（容差1e-12）；JVP方向导数比较差2.220e-16、JIT比较差0，CPU/complex128检查通过。未对导数施加正定性要求，因为它不是密度矩阵。
- 数据及软件/代码哈希元数据位于 [results/derivatives_20261014_154702_356659](results/derivatives_20261014_154702_356659/)，含参数、rho、AD/独立Jacobian及逐参数FD误差/mask。NPZ读回逐元素一致；检查脚本从项目外运行。命令见README。
- 从 `/private/tmp` 实际运行合并测试：`22 passed in 7.11s`；pip check通过。没有安装或修改依赖；先前Exact/ansatz核心代码和已验收结果均无改动；只在本子项目写输出。

## Phase 5：McLachlan 方程与 VQS 基线

- 从干净的Phase 4提交 `cd327b6` 创建本地 `phase5/vqs`；保留全部已有分支与历史。当前阶段提交编号以 `git log -1 --oneline phase5/vqs` 查询。
- 新增 `mclachlan.py`：当前约化rho的全部Lindblad项、D†内积M/V、虚部/对称性/PSD检查；SVD最小范数解显式阈值 `6*eps(float64)*s_max`。先说明阈值与×10/×100短时检查，再执行；无逆矩阵、Tikhonov、扰动或特殊首步。
- 初点M解析为(0,3)/(1,4)两块 `[[.5,1],[1,2]]`，奇异值约(2.5,2.5,1.17e-16,1.17e-16,0,0)，阈值3.33e-15，数值秩2；V=(1,1,0,2,2,0)，速度=(.4,.4,0,.8,.8,0)。完整条件数无穷，保留谱条件数1；初点切空间残差6.661e-16。
- 独立验证使用16×16门、门乘积法则导数、完整密度块部分迹、QuTiP Liouvillian和显式trace循环；17组参数M/V最大元素差1.443e-15/1.554e-15，容差1e-12。未编译/JIT差0；直接32×6实密度设计矩阵最小二乘对照速度最大差1.224e-10，容差1e-9；人工秩亏/零/全秩方程验证最小范数及不相容残差。
- `vqs.py`基线初参六零，参数Forward Euler dt=.01、t≤10；先通过短时检查才实际跑全程。每步更新前计算参数对应的rho与诊断，更新后从线路重建rho，保存1001点/1000次更新。没有读取Exact数据、rho-Euler、状态修复或改动物理方案。
- t≤.2的SVD阈值×10/×100结果与秩均未改变；dt=.01/.005和.005/.0025在共同时间点Z0最大差1.619e-4/8.365e-5。只作为短时诊断，不声称全时间收敛或阈值不敏感。
- 全程迹最大误差1.110e-15、厄米误差0、最小密度特征值0（物理容差1e-10）；参数/时间/观测量对齐通过（容差1e-12），11个跨全区间独立重建rho最大差1.943e-16。除初点秩2外1000点秩6；最大保留谱条件数3.173e8，最大参数步长范数.0314884。
- 全程最大线性残差5.759e-15，但最大切空间残差.0315028；后者是当前表示的生成元残差，不是观测量全时间误差。末点Z0=.1218784663。尚未与Exact计算误差，不能据此声称达到论文1e-2目标。
- [results/vqs_20261014_161232_948193](results/vqs_20261014_161232_948193/)保存 `vqs.npz`、`short_diagnostics.npz`与元数据（版本/单位/模型/阈值/源码及数据哈希）。两份NPZ读回逐元素一致；完整条件数Inf是有说明的秩亏标记，其余数值量有限。
- 从项目外 `/private/tmp` 实际运行全部测试：`29 passed in 7.87s`；pip check通过。没有安装或改变依赖；之前核心代码和已验收输出不变。运行命令见README，独立[中文公式与代码说明](notes/mclachlan_vqs_zh.md)已完成。

## Phase 6：Figure 5 与全点误差

- 从干净Phase 5提交 `3136aeb` 创建本地 `phase6/figure5`，保留全部历史；阶段哈希用 `git log -1 --oneline phase6/figure5` 查询。
- 重新目视核对原文第20页§6.3与第21页Figure 5，目标为Z0偏差小于1e-2，原图轴/图例/线点形式已核对。[来源记录](papers/figure5_comparison_source_zh.md)区分原文与本项目选择，明确插图减法次序在所核对正文/图注中未说明；项目按规划采用VQS−Exact。
- 新增 `plot.py` 与 `scripts/plot_figure5.py`，只读已验收Exact/VQS数据，不重跑动力学、不导入QuTiP/JAX/TensorCircuit。核实输入文件与元数据哈希、模型/基底、实际dtype与物理约束，并从rho独立核对Z0；两份1001点时间数组完全相同，无插值或坐标修正。
- 全点 `error=z0_vqs-z0_exact`；最大绝对误差0.0075584895671799435，n=256/t=2.56，该点误差为正。最小带符号差−0.006860966610591379，RMS=.003363036189909212，末点差−.0001371874208003776；严格1001点<.01目标满足。不声称作者原始数据逐点一致、全状态精度或连续时间严格上界。
- 仅主图VQS显示每15点抽一个并保留末点，共68标记；Exact线和误差插图用全部1001点。主范围t=0…10、Z0=−.2…1，插图范围±.009包含实际全部误差；没有翻转误差、平移时间、拟合或更改任何演化设定。
- 最终[results/figure5_20261014_162632_671057](results/figure5_20261014_162632_671057/)含1680×1296/240dpi `figure5.png`、全部比较值 `comparison.npz` 和来源/输入输出/源码哈希、版本与实际统计元数据。PNG已目视检查；NPZ已读回逐元素核对。输入原位保留，没有额外复制Exact/VQS轨迹。
- 7项新增测试验证独立trace循环、真实全点误差、被标记抽稀跳过的峰、.01严格边界及错位/损坏/低精度输入拒绝；从项目外运行全部测试实际结果 `36 passed in 7.97s`。绘图入口也从 `/private/tmp` 实际运行成功。
- 英文README新增默认及自选数据绘图命令、结果与限制；[中文比较说明](notes/figure5_comparison_zh.md)、学习入口与Git指南已更新。未安装或改动依赖，既有核心代码与已验收数据不变。

## 边界与后续限制

- 没有 push 或 merge，也没有通过连接器上传本地文件。
- /Users/ornfr/Desktop/bookandpaper 严格只读、处于 Git 根之外，未复制或链接进项目。
- `.venv`、缓存、临时日志均在本子项目并由现有 `.gitignore` 排除；只跟踪本阶段必要代码、文档及小型验证记录。
- 已完成全部六个实施阶段的代码、Exact/VQS数据、Figure 5主图/误差插图和1001点比较；达到指定观测量对照目标，等待用户最终验收。作者原始数组不可用，发表Exact线时间坐标差异仍按此前记录保留，无修正交付数据。
- 未运行上游完整测试套件、第二个全新环境重建或 GPU 检查。TensorCircuit 可提示缺少未使用的 Qiskit/Cirq/cloud/Optax；未为消除提示安装额外框架。环境依赖现有基础 Python 3.12 路径，迁移需重建并复验。
- 尚未做全时间步长/阈值收敛、完整状态误差或误差来源分离；当前停在Phase 6交付，不自动扩展研究。

## 下一步

等待用户选择：A验收并结束本次复现，保持本地成果；B先解释Figure 5误差插图及论文一致性的范围；C先提出全时间收敛检查的具体方案，等批准再计算。涉及改变初态、参数化、积分器或正则化仍须先给出证据并等待明确批准。本次不push、不merge，下一步选项也不隐含上传授权。
