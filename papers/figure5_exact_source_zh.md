# Figure 5 的 Exact 曲线：原文条件与比较边界

核对日期：2026-10-13（Asia/Shanghai）。此次仅完成 Exact 演化曲线，未实现完整 Figure 5 的变分点和误差插图。

## 实际阅读的来源

- Xiao Yuan, Suguru Endo, Qi Zhao, Ying Li, Simon C. Benjamin, *Theory of variational quantum simulation*, Quantum **3**, 191 (2019)。[期刊页面](https://quantum-journal.org/papers/q-2019-10-07-191/)，DOI: `10.22331/q-2019-10-07-191`。
- 只读本地文件：`/Users/ornfr/Desktop/bookandpaper/MS-VQD/Yuan et al. - 2019 - Theory of variational quantum simulation.pdf`。
- 41 页，SHA-256: `e142a56c57c822a45fb22aea7e7b9a47f694a9c37842796a040c9f8d4c2053b7`。该文件为期刊排版格式，页脚显示 accepted date 2019-09-07；不补写未经确认的 arXiv 版本号。资料未修改、复制或链接进 Git。
- 实际提取并目视核对：PDF 第 20 页（印刷页 20）的 §6.3、Eq. (62)、模型/初态；PDF 第 21 页（印刷页 21）的 Figure 5、坐标、图例和图注。页面截图与提取文本只在忽略的 `work/paper_review/`。

## 编码前确认的复现对象

| 项目 | 论文明确条件 |
| --- | --- |
| 系统 | 两量子比特 Ising 模型，独立振幅阻尼 |
| 哈密顿量 | $H=X_0+X_1+\tfrac14Z_0Z_1$ |
| 耗散 | $L_j(\rho)=\sigma_j^-\rho\sigma_j^+-\tfrac12\{\sigma_j^+\sigma_j^-,\rho\}$，$\sigma^-=|0\rangle\langle1|$ |
| 初态 | $\rho(0)=|00\rangle\langle00|$ |
| 时间和观测量 | $0\le t\le10$，$\langle Z_0\rangle$ |
| 数值方法说明 | 原文说明用 QuTiP 数值求解 Exact，给出 $\delta t=0.01$；未披露 Exact 求解器版本、算法、容差和原始数组 |
| 本次图片 | 只绘制蓝色 Exact 主曲线，横轴 0…10、纵轴 −0.2…1 |

本项目选择：$\hbar=1$，按横场系数定义无量纲时间；保存 1001 个点 $t_n=0.01n$，包含初态和末点；QuTiP 5.2.3 `mesolve`、自适应 DOP853、`atol=1e-12`、`rtol=1e-10`。这些求解器设置不声称是论文使用的具体设置。未归一化输出、未裁剪特征值、未平滑或拟合曲线。

## 原图矢量曲线的实际比较

`scripts/extract_figure5_reference.py` 使用已存在的 PDF 阅读运行时和 pdfplumber，读取 Figure 5 的蓝色折线（RGB 0.302,0.745,0.933），不是橙色变分点。它检查折线不是 Bezier 控制点，并核实主坐标框的四条边。框在 PDF 底部原点坐标中的端点为 x=(190.75963,414.83817)、y=(550.351725,726.413435)，对应图中 t=(0,10)、Z0=(−0.2,1)。pdfplumber 点坐标采用顶部原点，转换时已处理纵轴反向。

940 个实际绘图顶点及来源记录保存于 `papers/figure5_exact_reference.json`。这些顶点是发表图中的线条几何信息，不能称为作者提供的原始模拟数据。折线可能省略共线点；940 也不是作者原始时间网格点数。首个顶点为 `(t=0.01,Z0=1)`。

| 比较方法 | 最大绝对差 | 均方根差 |
| --- | --- | --- |
| 按原图顶点时间直接求解 | 0.011055198990139958 | 0.002900045299642943 |
| 对比较时间减去 0.01 | 0.0008709659381767321 | 0.0003070102171136438 |
| 比较时间采用 $(t_{plot}-0.01)1000/999$ | 1.790377870611315e-6 | 1.0148456543590459e-6 |

第三行的解释是假设原曲线使用 1000 个样本覆盖物理时间 0…10，却画在 $j\times0.01$（j=1…1000）上。它来自首个/末个坐标与等间隔样本编号的对应，不是通过改变 Hamiltonian、阻尼率或初态拟合出来。**这是能解释图形的采样坐标假设，未取得作者原始代码，不能断言作者确实如此实现。** 残余差异也不能无证据地归因于某个库版本或求解器容差。

各行都是在相应比较时间重新独立求解，而非对稀疏数据粗略插值。上述比较只用于说明发表线条的坐标差异；实际交付 `exact.npz` 和 PNG 始终使用初态在 t=0 的 1001 点网格，没有应用任何时间平移或比例调整。

## 可复查的证据与限制

最终数据、曲线、数值验证和软件记录：`results/exact_20261013_220047_446893/`。复算命令见 README。提取脚本只在重建图形参考时需要已有 pdfplumber；正常 `run_exact.py` 直接读取保存的参考 JSON，不向科研 `.venv` 新增 PDF 依赖。提取命令如下；文件已存在时会拒绝覆盖：

```bash
FIG5_ROOT='/Users/ornfr/Desktop/ai小垃圾/MSVQD/reproductions/yuan2019_fig5'
PYTHONPYCACHEPREFIX="$FIG5_ROOT/work/cache/pycache" \
  '/Users/ornfr/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3' \
  "$FIG5_ROOT/scripts/extract_figure5_reference.py" \
  --pdf '/Users/ornfr/Desktop/bookandpaper/MS-VQD/Yuan et al. - 2019 - Theory of variational quantum simulation.pdf'
```

完整 Figure 5 尚未复现；此次复现的是论文指定 Lindblad 模型的 Exact 演化，并透明记录与发表曲线坐标的比较。原文的“小于 1e-2”是 VQS 相对 Exact 的偏差，不能用它评价或替代本阶段的 ODE 精度检查或上述绘图坐标诊断。
