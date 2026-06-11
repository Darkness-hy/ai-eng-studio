# Python 环境管理

> 依赖地狱是真实存在的。虚拟环境就是解药。

**Type:** Build
**Languages:** Shell
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~30 minutes

## 学习目标

- 使用 `uv`、`venv` 或 `conda` 创建隔离的虚拟环境
- 编写带可选依赖组的 `pyproject.toml`，并生成锁文件（lockfile）以保证可复现性
- 诊断并修复常见陷阱：全局安装、pip 与 conda 混用、CUDA 版本不匹配
- 为依赖相互冲突的项目实施按阶段划分的环境策略

## 问题背景

你为一个微调项目安装了 PyTorch 2.4。一周后，另一个项目因为锁定了特定的 CUDA 构建版本，需要 PyTorch 2.1。你在全局升级，第一个项目坏了；你再降级，第二个项目又坏了。

这就是依赖地狱（dependency hell）。它在 AI/ML 工作中频繁发生，原因包括：

- PyTorch、JAX 和 TensorFlow 各自捆绑自己的 CUDA 绑定
- 模型库会锁定特定的框架版本
- 全局 `pip install` 会直接覆盖之前安装的版本
- CUDA 11.8 的构建无法在 CUDA 12.x 驱动上工作（反之亦然）

解决办法：每个项目都拥有自己独立的隔离环境和独立的软件包。

## 核心概念

```mermaid
graph TD
    subgraph without["Without virtual environments"]
        SP[System Python] --> T24["torch 2.4.0 (CUDA 12.4)\nProject A needs this"]
        SP --> T21["torch 2.1.0 (CUDA 11.8)\nProject B needs this"]
        SP --> CONFLICT["CONFLICT: only one\ntorch version can exist"]
    end

    subgraph with["With virtual environments"]
        PA["Project A (.venv/)"] --> PA1["torch 2.4.0 (CUDA 12.4)"]
        PA --> PA2["transformers 4.44"]
        PB["Project B (.venv/)"] --> PB1["torch 2.1.0 (CUDA 11.8)"]
        PB --> PB2["diffusers 0.28"]
    end
```

## 从零实现

### 方案 1：uv venv（推荐）

`uv` 是目前最快的 Python 包管理器（比 pip 快 10-100 倍）。它在一个工具中同时处理虚拟环境、Python 版本和依赖解析。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.12

cd your-project
uv venv
source .venv/bin/activate
```

安装软件包：

```bash
uv pip install torch numpy
```

一步创建带 `pyproject.toml` 的项目：

```bash
uv init my-ai-project
cd my-ai-project
uv add torch numpy matplotlib
```

### 方案 2：venv（内置）

如果你无法安装 `uv`，Python 自带 `venv`：

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

pip install torch numpy
```

比 `uv` 慢，但只要装了 Python 的地方就能用。

### 方案 3：conda（在需要时使用）

Conda 可以管理非 Python 依赖，比如 CUDA 工具包、cuDNN 和 C 库。在以下情况使用它：

- 你需要特定版本的 CUDA 工具包，但不想在系统全局安装
- 你在共享集群上工作，无法安装系统级软件包
- 某个库的安装说明明确写着"使用 conda"

```bash
# Install miniconda (not the full Anaconda)
curl -LsSf https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b

conda create -n myproject python=3.12
conda activate myproject

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

一条铁律：如果一个环境用 conda 创建，这个环境里的所有软件包就都用 conda 安装。在 conda 环境里混用 `pip install` 会导致难以调试的依赖冲突。

### 本课程采用：按阶段划分的环境策略

你完全可以为整个课程只建一个环境。但别这么做。不同阶段需要不同的（有时相互冲突的）依赖。

策略如下：

```
ai-engineering-from-scratch/
├── .venv/                    <-- shared lightweight env for phases 0-3
├── phases/
│   ├── 04-neural-networks/
│   │   └── .venv/            <-- PyTorch env
│   ├── 05-cnns/
│   │   └── .venv/            <-- same PyTorch env (symlink or shared)
│   ├── 08-transformers/
│   │   └── .venv/            <-- might need different transformer versions
│   └── 11-llm-apis/
│       └── .venv/            <-- API SDKs, no torch needed
```

`code/env_setup.sh` 中的脚本会为本课程创建基础环境。

## pyproject.toml 基础

每个 Python 项目都应该有一个 `pyproject.toml`。它用一个文件取代了 `setup.py`、`setup.cfg` 和 `requirements.txt`。

```toml
[project]
name = "ai-engineering-from-scratch"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "matplotlib>=3.8",
    "jupyter>=1.0",
    "scikit-learn>=1.4",
]

[project.optional-dependencies]
torch = ["torch>=2.3", "torchvision>=0.18"]
llm = ["anthropic>=0.39", "openai>=1.50"]
```

然后安装：

```bash
uv pip install -e ".[torch]"    # base + PyTorch
uv pip install -e ".[llm]"     # base + LLM SDKs
uv pip install -e ".[torch,llm]" # everything
```

## 锁文件

锁文件（lockfile）把每一个依赖（包括传递依赖）都钉死在精确版本上。这保证了可复现性：任何人从锁文件安装，得到的软件包都完全一致。

```bash
# uv generates uv.lock automatically when using uv add
uv add numpy

# pip-tools approach
uv pip compile pyproject.toml -o requirements.lock
uv pip install -r requirements.lock
```

把锁文件提交到 git。当别人克隆仓库后，从锁文件安装就能得到一模一样的版本。

## 常见错误

### 1. 全局安装

```bash
pip install torch  # BAD: installs to system Python

source .venv/bin/activate
pip install torch  # GOOD: installs to virtual environment
```

检查软件包安装到了哪里：

```bash
which python       # should show .venv/bin/python, not /usr/bin/python
which pip           # should show .venv/bin/pip
```

### 2. pip 和 conda 混用

```bash
conda create -n myenv python=3.12
conda activate myenv
conda install pytorch -c pytorch
pip install some-other-package   # BAD: can break conda's dependency tracking
conda install some-other-package # GOOD: let conda manage everything
```

如果你确实必须在 conda 里用 pip（有些包只能通过 pip 安装），先装完所有 conda 包，最后再装 pip 包。

### 3. 忘记激活环境

```bash
python train.py           # uses system Python, missing packages
source .venv/bin/activate
python train.py           # uses project Python, packages found
```

你的 shell 提示符应该显示环境名：

```
(.venv) $ python train.py
```

### 4. 把 .venv 提交到 git

```bash
echo ".venv/" >> .gitignore
```

虚拟环境的体积通常在 200MB-2GB 之间。它们只在本机有效，不能在机器之间迁移。应该提交的是 `pyproject.toml` 和锁文件。

### 5. CUDA 版本不匹配

```bash
nvidia-smi                # shows driver CUDA version (e.g., 12.4)
python -c "import torch; print(torch.version.cuda)"  # shows PyTorch CUDA version

# These must be compatible.
# PyTorch CUDA version must be <= driver CUDA version.
```

## 生产实践

运行安装脚本，创建本课程的环境：

```bash
bash phases/00-setup-and-tooling/06-python-environments/code/env_setup.sh
```

它会在仓库根目录创建一个 `.venv`，安装核心依赖并完成验证。

## 练习

1. 运行 `env_setup.sh`，确认所有检查项都通过
2. 创建第二个虚拟环境，在其中安装不同版本的 numpy，确认两个环境互相隔离
3. 为一个同时需要 PyTorch 和 Anthropic SDK 的项目编写 `pyproject.toml`
4. 故意在全局安装一个软件包（不激活任何 venv），观察它被装到了哪里，然后卸载它

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|----------------------|
| 虚拟环境（Virtual environment） | "一个 venv" | 一个包含 Python 解释器和软件包的隔离目录，与系统 Python 相互独立 |
| 锁文件（Lockfile） | "钉死的依赖" | 列出每个软件包及其精确版本的文件，保证在不同机器上安装结果完全一致 |
| pyproject.toml | "新版 setup.py" | Python 项目的标准配置文件，取代 setup.py/setup.cfg/requirements.txt |
| 传递依赖（Transitive dependency） | "依赖的依赖" | 包 B 依赖 C；当你安装依赖 B 的包 A 时，C 就是 A 的传递依赖 |
| CUDA 不匹配（CUDA mismatch） | "我的 GPU 不工作了" | PyTorch 编译时使用的 CUDA 版本与你的 GPU 驱动支持的版本不一致 |
