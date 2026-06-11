# 编辑器配置

> 编辑器是你的副驾驶。一次性配置好，它就不会碍手碍脚，并开始为你分担工作。

**Type:** Build
**Languages:** --
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~20 minutes

## 学习目标

- 安装 VS Code 及 Python、Jupyter、代码检查（linting）和远程 SSH 所需的核心扩展
- 配置保存时自动格式化、类型检查和笔记本输出滚动，以适配 AI 工作流
- 配置 Remote SSH，让你在远程 GPU 机器上编辑和调试代码时如同在本地一样
- 评估其他编辑器选项（Cursor、Windsurf、Neovim）及其在 AI 工作中的取舍

## 问题背景

你会在编辑器里度过成千上万个小时：写 Python、跑笔记本、调试训练循环、SSH 登录 GPU 机器。配置不当的编辑器会让每次工作都充满摩擦：没有自动补全、没有类型提示、没有内联错误提示、需要手动格式化，终端工作流也很笨拙。

正确的配置只需要 20 分钟。跳过它，每天都要多付出 20 分钟。

## 核心概念

一套面向 AI 工程的编辑器配置需要五样东西：

```mermaid
graph TD
    L5["5. Remote Development<br/>SSH into GPU boxes, cloud VMs"] --> L4
    L4["4. Terminal Integration<br/>Run scripts, debug, monitor GPU"] --> L3
    L3["3. AI-Specific Settings<br/>Auto-format, type checking, rulers"] --> L2
    L2["2. Extensions<br/>Python, Jupyter, Pylance, GitLens"] --> L1
    L1["1. Base Editor<br/>VS Code — free, extensible, universal"]
```

## 从零实现

### 第 1 步：安装 VS Code

VS Code 是推荐的编辑器。它免费、支持所有操作系统、对 Jupyter 笔记本有一流支持，扩展生态覆盖了 AI 工作所需的一切。

从 [code.visualstudio.com](https://code.visualstudio.com/) 下载。

在终端中验证：

```bash
code --version
```

如果在 macOS 上找不到 `code` 命令，打开 VS Code，按 `Cmd+Shift+P`，输入 "Shell Command"，然后选择 "Install 'code' command in PATH"。

### 第 2 步：安装核心扩展

打开 VS Code 的集成终端（`Ctrl+`` ` 或 `` Cmd+` ``），安装对 AI 工作至关重要的扩展：

```bash
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter
code --install-extension eamodio.gitlens
code --install-extension ms-vscode-remote.remote-ssh
code --install-extension ms-python.debugpy
code --install-extension ms-python.black-formatter
code --install-extension charliermarsh.ruff
```

各扩展的作用：

| 扩展 | 作用 |
|-----------|-----|
| Python | 语言支持、虚拟环境检测、运行/调试 |
| Pylance | 快速类型检查、自动补全、导入解析 |
| Jupyter | 在 VS Code 内运行笔记本、变量浏览器 |
| GitLens | 查看谁改了什么、内联 git blame |
| Remote SSH | 像本地一样打开远程 GPU 机器上的文件夹 |
| Debugpy | Python 单步调试 |
| Black Formatter | 保存时自动格式化、风格统一 |
| Ruff | 快速代码检查，捕获常见错误 |

本课的 `code/.vscode/extensions.json` 文件包含完整的推荐列表。打开项目文件夹时，VS Code 会提示你安装它们。

### 第 3 步：配置设置

复制本课 `code/.vscode/settings.json` 中的设置，或通过 `Settings > Open Settings (JSON)` 手动应用。

对 AI 工作最关键的几项设置：

```jsonc
{
    "python.analysis.typeCheckingMode": "basic",
    "editor.formatOnSave": true,
    "editor.rulers": [88, 120],
    "notebook.output.scrolling": true,
    "files.autoSave": "afterDelay"
}
```

这些设置为何重要：

- **类型检查设为 basic**：在运行之前就捕获错误的参数类型。在张量形状不匹配和 API 参数传错的问题上为你省下调试时间。
- **保存时格式化**：再也不用操心格式问题。交给 Black 处理。
- **88 和 120 处的标尺**：Black 在 88 列处换行。120 的标线提示你 docstring 和注释是否写得太长了。
- **笔记本输出滚动**：训练循环会打印成千上万行输出。不开滚动，输出面板会爆掉。
- **自动保存**：你总会忘记保存。训练脚本就会跑在过期的代码上。自动保存能避免这种情况。

### 第 4 步：终端集成

VS Code 的集成终端是你运行训练脚本、监控 GPU、管理环境的地方。

把它配置好：

```jsonc
{
    "terminal.integrated.defaultProfile.osx": "zsh",
    "terminal.integrated.defaultProfile.linux": "bash",
    "terminal.integrated.fontSize": 13,
    "terminal.integrated.scrollback": 10000
}
```

常用快捷键：

| 操作 | macOS | Linux/Windows |
|--------|-------|---------------|
| 切换终端 | `` Ctrl+` `` | `` Ctrl+` `` |
| 新建终端 | `Ctrl+Shift+`` ` | `Ctrl+Shift+`` ` |
| 拆分终端 | `Cmd+\` | `Ctrl+\` |

拆分终端很实用：一个窗格运行脚本，另一个用 `nvidia-smi -l 1` 或 `watch -n 1 nvidia-smi` 监控 GPU。

### 第 5 步：远程开发（SSH 登录 GPU 机器）

这是 AI 工作中最重要的扩展。你会在远程机器上跑训练（云虚拟机、实验室服务器、Lambda、Vast.ai）。Remote SSH 让你打开远程文件系统、编辑文件、运行终端、进行调试，一切都像在本地一样。

配置步骤：

1. 安装 Remote SSH 扩展（第 2 步已完成）。
2. 按 `Ctrl+Shift+P`（或 `Cmd+Shift+P`），输入 "Remote-SSH: Connect to Host"。
3. 输入 `user@your-gpu-box-ip`。
4. VS Code 会自动在远程机器上安装其服务端组件。

要实现免密登录，配置 SSH 密钥：

```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
ssh-copy-id user@your-gpu-box-ip
```

为方便起见，把主机加入 `~/.ssh/config`：

```
Host gpu-box
    HostName 203.0.113.50
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519
    ForwardAgent yes
```

之后通过 `Remote-SSH: Connect to Host > gpu-box` 即可秒连。

## 其他选择

### Cursor

[cursor.com](https://cursor.com) 是一个内置 AI 代码生成的 VS Code 分支。它使用相同的扩展生态和设置格式。如果你用 Cursor，本课的所有内容仍然适用。直接导入同样的 `settings.json` 和 `extensions.json` 即可。

### Windsurf

[windsurf.com](https://windsurf.com) 是另一个 AI 优先的 VS Code 分支。情况相同：同样的扩展、同样的设置格式、同样支持 Remote SSH。

### Vim/Neovim

如果你已经在用 Vim 或 Neovim 并且用得很顺手，就继续用。面向 AI Python 工作的最小配置：

- **pyright** 或 **pylsp** 做类型检查（通过 Mason 或手动安装）
- **nvim-lspconfig** 做语言服务器集成
- **jupyter-vim** 或 **molten-nvim** 实现类似笔记本的执行体验
- **telescope.nvim** 做文件/符号搜索
- **none-ls.nvim** 搭配 black 和 ruff 做格式化/代码检查

如果你还没在用 Vim，现在别开始学。它的学习曲线会和学习 AI 工程争夺时间。用 VS Code。

## 生产实践

配置完成后，你的日常工作流是这样的：

1. 在 VS Code 中打开项目文件夹（或通过 Remote SSH 连接到 GPU 机器）。
2. 在编辑器里写 Python，享受自动补全、类型提示和内联错误提示。
3. 用 Jupyter 扩展在编辑器内运行 Jupyter 笔记本。
4. 用集成终端运行训练脚本、执行 `uv pip install`、监控 GPU。
5. 提交前用 GitLens 审查改动。

## 练习

1. 安装 VS Code 及第 2 步列出的所有扩展
2. 把本课的 `settings.json` 复制到你的 VS Code 配置中
3. 打开一个 Python 文件，验证 Pylance 显示类型提示、Black 在保存时格式化
4. 如果你有远程机器可用，配置 Remote SSH 并在其上打开一个文件夹

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|----------------------|
| LSP | “自动补全引擎” | 语言服务器协议（Language Server Protocol）：一种标准，让编辑器从特定语言的服务器获取类型信息、补全和诊断 |
| Pylance | “那个 Python 插件” | 微软的 Python 语言服务器，基于 Pyright 提供类型检查和 IntelliSense |
| Remote SSH | “在服务器上干活” | 一个 VS Code 扩展，在远程机器上运行轻量服务端，把界面串流到你本地的编辑器 |
| 保存时格式化 | “自动 prettier” | 编辑器在每次保存时运行格式化工具（Black、Ruff），让代码风格始终一致 |
