# Figure 5 reproduction progress

日期：2026-10-13（Asia/Shanghai）。用户在 Phase 1 交付后明确要求实现 Exact 曲线；Phase 2 的 Exact 计算、验证与绘图已完成，停下等待下一步指示。

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

## 边界与未开始工作

- 没有 push 或 merge，也没有通过连接器上传本地文件。
- /Users/ornfr/Desktop/bookandpaper 严格只读、处于 Git 根之外，未复制或链接进项目。
- `.venv`、缓存、临时日志均在本子项目并由现有 `.gitignore` 排除；只跟踪本阶段必要代码、文档及小型验证记录。
- 只实现 Exact 演化并按本次请求绘制其曲线；未实现 purification、部分迹、McLachlan 或 VQS 演化，未完成含变分点/误差插图的完整 Figure 5。
- 未运行上游完整测试套件、第二个全新环境重建或 GPU 检查。TensorCircuit 可提示缺少未使用的 Qiskit/Cirq/cloud/Optax；未为消除提示安装额外框架。环境依赖现有基础 Python 3.12 路径，迁移需重建并复验。
- 当前停在 Phase 2 的 Exact 交付；不自动进入后续阶段。

## 下一步

等待用户对 Exact 曲线的反馈和下一步明确指示。后续 Figure 4 线路、purification、AD/McLachlan/VQS 均未获本次授权；本次不 push、不 merge。
