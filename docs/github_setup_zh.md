# GitHub 仓库与 Mac 本地关联指南

更新日期：2026-10-13。

## 已完成的关联

- 私有仓库：[ORnFr/yuan2019-fig5](https://github.com/ORnFr/yuan2019-fig5)。
- 本地项目：`/Users/ornfr/Desktop/ai小垃圾/MSVQD/reproductions/yuan2019_fig5/`。
- origin：`https://github.com/ORnFr/yuan2019-fig5.git`。
- 本地 Git 认证已实际验证：fetch 成功，不再需要重复登录。
- 本地 main 跟踪 origin/main；两者以远程初始 README 提交 `e7927b483d7d83b0e67a86df16ab009e89eadb97` 为共同基线。
- 规划分支：`phase0/planning`；规划文档提交只保存在本地。
- 经用户确认，仓库内作者为 `ORnFr <307864480+ORnFr@users.noreply.github.com>`；没有更改全局 Git 身份。
- 没有 push、merge 或上传本地文件。GitHub 页面目前仍只有初始 README，这是正常状态。

## 查看当前状态

```bash
cd '/Users/ornfr/Desktop/ai小垃圾/MSVQD/reproductions/yuan2019_fig5'
git status --short --branch
git branch -vv
git log --oneline --decorate -3
git remote -v
```

Phase 1 环境提交为 `d9d208d`。随后用户要求实现 Exact 曲线，当前阶段分支为 `phase2/exact`，从该环境提交继续开发。阶段提交后工作区应无待提交改动。main 仍跟踪 origin/main；规划、环境和 Exact 分支均只保存在本地，这是未 push 的正常状态。

如需检查仓库范围：

```bash
git rev-parse --show-toplevel
git ls-files
```

Git 根应精确返回本子项目路径。跟踪清单可含规划文档、规范、环境检查脚本、测试、固定依赖与小型验证记录，不能含 `.venv`、`work`、父工作区或论文。

## 后续本地阶段提交

每阶段通过科学验证且完成验收说明后，再显式添加本阶段修改文件。以下只是后续操作模板，请先将路径替换为真实修改文件，不直接执行占位符：

```text
git add <本阶段的具体文件路径>
git diff --cached --stat
git diff --cached
git commit -m '<描述本阶段成果的提交说明>'
```

不要在 MSVQD 或 Desktop 初始化 Git，也不要在父目录执行 git add .。当前已正确建立远程基线，不需要再初始化、重置分支或合并不相关历史。不要重跑旧指南的初始化命令。

## 文献与输出边界

实际文献目录是 `/Users/ornfr/Desktop/bookandpaper/`，在项目根之外，严格只读。不要复制论文或建立指向文献的链接；使用 DOI/arXiv/期刊链接引用。缓存、.venv、临时 work 目录和外来 PDF 已忽略；后续最终 NPZ/JSON 数据及 Figure 5 PNG 可以跟踪。

## 下一步

审阅项目 PROGRESS.md、README 和 Exact 结果。Phase 2 的 Exact 演化、曲线及验证已完成，当前停下等待下一步指示。未实现 VQS，不自动进入后续阶段。未经单独批准不 push 或 merge。
