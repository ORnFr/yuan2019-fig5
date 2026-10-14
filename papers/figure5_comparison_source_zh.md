# Phase 6：Figure 5 对照目标与来源边界

日期：2026-10-14（Asia/Shanghai）。本阶段重新目视检查项目 `work/paper_review/` 中此前由只读论文渲染的第20/21页。原件未修改、复制或链接进Git。

来源：Xiao Yuan, Suguru Endo, Qi Zhao, Ying Li, Simon C. Benjamin，*Theory of variational quantum simulation*，Quantum **3**, 191 (2019)，[期刊页面](https://quantum-journal.org/papers/q-2019-10-07-191/)，DOI `10.22331/q-2019-10-07-191`。本地41页期刊排版PDF的SHA-256为 `e142a56c57c822a45fb22aea7e7b9a47f694a9c37842796a040c9f8d4c2053b7`，页脚accepted date为2019-09-07；不推断arXiv版本号。

- PDF/印刷第20页 §6.3、Eq.(62) 给出模型、初态及六零初参、dt=.01；末段报告 Z0 的 Exact/变分偏差小于 $10^{-2}$。本项目以保存网格上最大绝对误差严格小于.01作为可检验对照；不是声称论文明确使用了本项目相同的统计程序。
- PDF/印刷第21页 Figure 5 主图是Exact线与变分点；横轴0…10，纵轴−.2…1。插图是带正负的Error；本次核对的正文/图注没有明确给出减法次序。本项目按已批准规划定义“VQS − Exact”，在图轴上直接写明，不能通过翻转符号贴合发表插图。
- 第21页图注说明dt=.01、t=0…10及QuTiP数值Exact。原始数组、SVD阈值、完整积分实现和软件版本没有随该图提供；本项目的具体配置属于明确记录的选择。

本次实际达到的目标是所规定模型、Figure 4线路和项目基线下的Z0对比及1001点最大差0.00755849。没有数字化作者VQS点或误差插图做逐点匹配，不称完整原始数组复现。原图Exact线的坐标差异继续参考[此前来源记录](figure5_exact_source_zh.md)；最终图片未应用时间平移或比例调整。

输出、带符号误差定义、独立核对和使用说明见[中文比较说明](../notes/figure5_comparison_zh.md)与[本阶段运行元数据](../results/figure5_20261014_162632_671057/run_metadata.json)。
