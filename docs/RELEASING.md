# 首次发布与后续更新

本次使用全新 Git 历史发布源码快照。原工作目录保留本机依赖、数据和历史；
它曾含有硬编码 API Key，不能通过普通 commit 删除后直接推送旧分支来解决历史泄露。
先在服务提供方吊销/轮换旧 Key，并在本机使用新 Key。
生成干净快照不会清理旧远端、旧克隆或备份中的历史。

## 导出（由维护者在原工作目录执行）

```bash
python3 scripts/prepare_release.py --output /absolute/path/to/armcode-agent-source-release
```

目标目录必须不存在，且必须位于当前源码目录之外。导出器包含当前工作树的新增源码，
不会仅导出旧的已提交版本；导出前应确认这些修改正是本次要发布的版本。
它采用目录和类型白名单，排除 `.git`、模型、外部 SDK、授权文件、本机配置、
图库、实验数据和运行产物，生成 `RELEASE_MANIFEST.json`（导出时文件的 SHA-256）。
匹配到常见凭据格式时会停止，提示位置但不打印凭据；模式扫描不等于完整安全审计。
修改发布范围时同步修改导出器和回归检查。

## 验证与发布（在导出的目录中执行）

按 [SETUP](SETUP.md) 建立独立环境，运行：

```bash
python scripts/prepare_release.py --check
python scripts/check_core.py
python -m app --help
```

在 GitHub 新建一个空仓库，首次协作建议选择 Private，不预创建 README、LICENSE 或 `.gitignore`。
将下面的地址替换成**新仓库** SSH 地址，在导出目录执行：

```bash
git init -b main
git add .
python scripts/prepare_release.py --check-tracked
git diff --cached --stat
git commit -m "Prepare collaborative source baseline"
git remote add origin git@github.com:YOUR_ACCOUNT/YOUR_NEW_REPOSITORY.git
git push -u origin main
git tag -a v0.1.0 -m "First collaborative source baseline"
git push origin v0.1.0
```

`v0.1.0` 是首个源码协作基线标签，不代表所有设备和模型在任意电脑开箱即用。
这些命令推送代码和标签，不自动创建 GitHub Releases 页面。
不要在原目录执行上述初始化命令，也不要推送旧分支、使用 `--mirror` 或 force push 覆盖原仓库。

如果复用旧仓库名字，可将原仓库保留/归档后另行决定迁移；本次流程不改写任何旧历史。
本机原始配置和模型仍在原工作目录中，不应再复制进新仓库的提交。

## 发布后进入正常协作

所有成员从新仓库克隆，按 [CONTRIBUTING](../CONTRIBUTING.md) 使用任务分支和 PR。
管理员配置 `main` 保护与 CI 要求；仓库文件不会自动设置这些服务端权限。
集成电脑也从新仓库建一个工作副本，并单独复制本机配置、关联外部模型环境。
首次验证后再切换日常运行目录，保留原目录作回退。

后续更新正常提交到新仓库，不要每次重建历史；导出器只在需要生成独立源码包时使用。
`RELEASE_MANIFEST.json` 是首次导出的快照记录，不是随每次修改自动更新的锁文件。
本地使用的 GitHub Actions 结构参考 [官方 Python CI 文档](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)。
