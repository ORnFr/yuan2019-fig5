# Figure 4 线路核对记录

日期：2026-10-14（Asia/Shanghai）。只实现 Phase 3 的线路与部分迹，不实现参数导数、McLachlan 方程或 VQS 演化。

## 原文与实际核对

来源：Xiao Yuan, Suguru Endo, Qi Zhao, Ying Li, Simon C. Benjamin, *Theory of variational quantum simulation*, Quantum **3**, 191 (2019)。[期刊页面](https://quantum-journal.org/papers/q-2019-10-07-191/)，DOI `10.22331/q-2019-10-07-191`。

本地只读 PDF 的完整出处、版本边界与哈希见 [Exact 来源记录](figure5_exact_source_zh.md)。此次重新目视检查 `work/paper_review/yuan-20.png`（之前由该 PDF 第 20 页生成），核对印刷/PDF 第 20 页 Figure 4 和 §6.3。没有复制、修改或链接原始论文；该 PDF 的 SHA-256 为 `e142a56c57c822a45fb22aea7e7b9a47f694a9c37842796a040c9f8d4c2053b7`。

原文四行从上到下为辅助位 0′、1′，系统位 0、1；受控门的实心控制点在辅助位，控制值为 1。从左向右的操作为：辅助位 H，两个受控 RX，系统 RZZ，两个系统 RX，最后系统 RZZ。六个参数沿用论文编号 theta0…theta5。

原文定义：$R_X(\theta)=e^{-i\theta X}$，$R_{ZZ}(\theta)=e^{-i\theta Z\otimes Z}$。本项目将原文位标签映射为 `(a0,a1,s0,s1)=(0,1,2,3)`；这是软件位序选择，不是新增物理条件。系统的 $Z_0$ 对应 wire 2，和 Exact 模块的左侧系统张量因子对应。

## 已安装库的逐门核对

没有仅凭 API 名称猜测角度。实际检查了安装的 TensorCircuit 0.12.0 `gates.py`，并保存其文件 SHA-256 到验证元数据。参考 [官方 gate 文档](https://tensorcircuit.readthedocs.io/en/latest/api/gates.html)；文档可能随版本更新，本次实现依据固定安装版本的源码与数值比较。

| 原文门 | 安装版本实现 | 本项目调用 |
| --- | --- | --- |
| RX(theta) | `rx_gate`（第579行附近）为 exp(−i angle X/2) | `rx(...,theta=2*theta)` |
| CRX(theta) | `controlled`（第294行附近）将 RX 嵌入 control=1 的矩阵块；动态注册 crx（第968行附近） | `crx(control,target,theta=2*theta)` |
| RZZ(theta) | `rzz_gate`（第863行附近）绑定 `exp1_gate(...,half=True)`；half 会将 angle 除以2 | `rzz(...,theta=2*theta,half=True)` |

矩阵测试使用 SciPy `expm` 独立构造 paper RX、块对角 CRX、RZZ，对角度 0、0.17、−0.41、0.73、π/2 的完整矩阵和全部输入基矢比较。control=0 的目标没有变化；控制方向和所有矩阵列都通过。额外检查在 angle=0.37 时，少乘2会造成约0.178的矩阵元素差，说明选定非零角度能检测该错误。门最大比较误差约1.11e-16，容差1e-12。

## 实现与适用边界

核心函数见 `src/yuan2019_fig5/ansatz.py`。部分迹按已核对的辅助/系统分组进行，与完整16×16密度矩阵的显式块求和、解析态和 QuTiP `ptrace([2,3])` 独立比较。零参数的系统初态符合论文，但整体四位态因为辅助位的 H 已经变为 $|++\rangle_A|00\rangle_S$，不是 $|0000\rangle$。

最终验证及17组参数样本位于 `results/ansatz_20261014_152519_728727/`。随机参数种子2019、额外部分迹随机态种子2026，全部检查容差和实际误差在 `run_metadata.json`。样本不是作者原始数据，也不是 VQS 时间轨迹；未读取 Exact 数据拟合参数。
