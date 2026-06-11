# 终端与 Shell

> 终端是 AI 工程师安身立命的地方。先在这里练到得心应手。

**Type:** Learn
**Languages:** --
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~35 minutes

## 学习目标

- 使用管道（pipe）、重定向和 `grep` 在命令行中过滤、处理训练日志
- 创建带多个面板的持久化 tmux 会话，同时跑训练和监控 GPU
- 用 `htop`、`nvtop` 和 `nvidia-smi` 监控系统与 GPU 资源
- 使用 SSH、`scp` 和 `rsync` 在本地与远程机器之间传输文件

## 问题背景

你在终端里花的时间会比在任何编辑器里都多。跑训练、监控 GPU、追踪日志、远程 SSH 会话、管理环境——每一条 AI 工作流都绕不开 shell。在这里慢，你做什么都慢。

这节课只讲对 AI 工作真正有用的终端技能。不讲 Unix 历史，不深入 Bash 脚本编程，只讲你需要的东西。

## 核心概念

```mermaid
graph TD
    subgraph tmux["tmux session: training"]
        subgraph top["Top row"]
            P1["Pane 1: Training run<br/>python train.py<br/>Epoch 12/100 ..."]
            P2["Pane 2: GPU monitor<br/>watch -n1 nvidia-smi<br/>GPU: 78% | Mem: 14/24G"]
        end
        P3["Pane 3: Logs + experiments<br/>tail -f logs/train.log | grep loss"]
    end
```

三件事同时运行，只用一个终端。你可以断开会话（detach）、回家、再 SSH 回来重新连上（reattach），训练一直在跑。

## 从零实现

### 第 1 步：认识你的 shell

查看当前使用的 shell：

```bash
echo $SHELL
```

大多数系统用 `bash` 或 `zsh`，两者都没问题。本课程中的命令在两种 shell 里都能用。

需要掌握的要点：

```bash
# Move around
cd ~/projects/ai-engineering-from-scratch
pwd
ls -la

# History search (most useful shortcut you'll learn)
# Ctrl+R then type part of a previous command
# Press Ctrl+R again to cycle through matches

# Clear terminal
clear   # or Ctrl+L

# Cancel a running command
# Ctrl+C

# Suspend a running command (resume with fg)
# Ctrl+Z
```

### 第 2 步：管道与重定向

管道把多个命令串联起来。处理日志、过滤输出、组合各种工具，靠的就是它。你会一直用到。

```bash
# Count how many times "loss" appears in a log
cat train.log | grep "loss" | wc -l

# Extract just the loss values from training output
grep "loss:" train.log | awk '{print $NF}' > losses.txt

# Watch a log file update in real time, filtering for errors
tail -f train.log | grep --line-buffered "ERROR"

# Sort experiments by final accuracy
grep "final_accuracy" results/*.log | sort -t= -k2 -n -r

# Redirect stdout and stderr to separate files
python train.py > output.log 2> errors.log

# Redirect both to the same file
python train.py > train_full.log 2>&1
```

你需要的几个重定向符号：

| 符号 | 作用 |
|--------|-------------|
| `>` | 将 stdout 写入文件（覆盖） |
| `>>` | 将 stdout 追加到文件 |
| `2>` | 将 stderr 写入文件 |
| `2>&1` | 将 stderr 发送到与 stdout 相同的位置 |
| `\|` | 将前一个命令的 stdout 作为下一个命令的 stdin |

### 第 3 步：后台进程

训练一跑就是几个小时，你不会想一直开着终端。

```bash
# Run in background (output still goes to terminal)
python train.py &

# Run in background, immune to hangup (closing terminal won't kill it)
nohup python train.py > train.log 2>&1 &

# Check what's running in background
jobs
ps aux | grep train.py

# Bring a background job to foreground
fg %1

# Kill a background process
kill %1
# or find its PID and kill that
kill $(pgrep -f "train.py")
```

`&`、`nohup` 和 `screen`/`tmux` 的区别：

| 方式 | 关闭终端后还能存活？ | 能重新接入？ |
|--------|-------------------------|---------------|
| `command &` | 否 | 否 |
| `nohup command &` | 是 | 否（查看日志文件） |
| `screen` / `tmux` | 是 | 是 |

凡是超过几分钟的任务，用 tmux。

### 第 4 步：tmux

tmux 可以创建带多个面板的持久化终端会话。在管理训练任务这件事上，它是最有用的一个工具。

```bash
# Install
# macOS
brew install tmux
# Ubuntu
sudo apt install tmux

# Start a named session
tmux new -s training

# Split horizontally
# Ctrl+B then "

# Split vertically
# Ctrl+B then %

# Navigate between panes
# Ctrl+B then arrow keys

# Detach (session keeps running)
# Ctrl+B then d

# Reattach
tmux attach -t training

# List sessions
tmux ls

# Kill a session
tmux kill-session -t training
```

一个典型的 AI 工作流会话：

```bash
tmux new -s train

# Pane 1: start training
python train.py --epochs 100 --lr 1e-4

# Ctrl+B, " to split, then run GPU monitor
watch -n1 nvidia-smi

# Ctrl+B, % to split vertically, tail the logs
tail -f logs/experiment.log

# Now detach with Ctrl+B, d
# SSH out, go get coffee, come back
# tmux attach -t train
```

### 第 5 步：用 htop 和 nvtop 监控

```bash
# System processes (better than top)
htop

# GPU processes (if you have NVIDIA GPU)
# Install: sudo apt install nvtop (Ubuntu) or brew install nvtop (macOS)
nvtop

# Quick GPU check without nvtop
nvidia-smi

# Watch GPU usage update every second
watch -n1 nvidia-smi

# See which processes are using the GPU
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
```

你会用到的 `htop` 快捷键：
- `F6` 或 `>`：按列排序（按内存排序可以找出内存泄漏）
- `F5`：切换树状视图（查看子进程）
- `F9`：杀掉一个进程
- `/`：按名称搜索进程

### 第 6 步：用 SSH 连接远程 GPU 机器

当你租用云 GPU（Lambda、RunPod、Vast.ai）时，要通过 SSH 连接。

```bash
# Basic connection
ssh user@gpu-box-ip

# With a specific key
ssh -i ~/.ssh/my_gpu_key user@gpu-box-ip

# Copy files to remote
scp model.pt user@gpu-box-ip:~/models/

# Copy files from remote
scp user@gpu-box-ip:~/results/metrics.json ./

# Sync a whole directory (faster for many files)
rsync -avz ./data/ user@gpu-box-ip:~/data/

# Port forward (access remote Jupyter/TensorBoard locally)
ssh -L 8888:localhost:8888 user@gpu-box-ip
# Now open localhost:8888 in your browser

# SSH config for convenience
# Add to ~/.ssh/config:
# Host gpu
#     HostName 192.168.1.100
#     User ubuntu
#     IdentityFile ~/.ssh/gpu_key
#
# Then just:
# ssh gpu
```

### 第 7 步：AI 工作常用别名

把这些加进你的 `~/.bashrc` 或 `~/.zshrc`：

```bash
source phases/00-setup-and-tooling/10-terminal-and-shell/code/shell_aliases.sh
```

或者只挑你想要的复制过去。关键别名：

```bash
# GPU status at a glance
alias gpu='nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader'

# Kill all Python training processes
alias killtraining='pkill -f "python.*train"'

# Quick virtual environment activate
alias ae='source .venv/bin/activate'

# Watch training loss
alias watchloss='tail -f logs/*.log | grep --line-buffered "loss"'
```

完整列表见 `code/shell_aliases.sh`。

### 第 8 步：AI 终端常见模式

这些模式在实践中会反复出现：

```bash
# Run training, log everything, notify when done
python train.py 2>&1 | tee train.log; echo "DONE" | mail -s "Training complete" you@email.com

# Compare two experiment logs side by side
diff <(grep "accuracy" exp1.log) <(grep "accuracy" exp2.log)

# Find the largest model files (clean up disk space)
find . -name "*.pt" -o -name "*.safetensors" | xargs du -h | sort -rh | head -20

# Download a model from Hugging Face
wget https://huggingface.co/model/resolve/main/model.safetensors

# Untar a dataset
tar xzf dataset.tar.gz -C ./data/

# Count lines in all Python files (see how big your project is)
find . -name "*.py" | xargs wc -l | tail -1

# Check disk space (training data fills disks fast)
df -h
du -sh ./data/*

# Environment variable check before training
env | grep -i cuda
env | grep -i torch
```

## 生产实践

本课程中各工具的使用场景：

| 工具 | 使用时机 |
|------|----------------|
| tmux | 每一次训练（Phase 3 及以后） |
| `tail -f` + `grep` | 监控训练日志 |
| `nohup` / `&` | 临时的后台任务 |
| `htop` / `nvtop` | 排查训练变慢、OOM 错误 |
| SSH + `rsync` | 在云 GPU 上工作 |
| 管道 + 重定向 | 处理实验结果 |
| 别名 | 给重复性命令省时间 |

## 练习

1. 安装 tmux，创建一个含三个面板的会话：一个跑 `htop`，一个跑 `watch -n1 date`，第三个跑一个 Python 脚本。断开会话再重新接入。
2. 把 `code/shell_aliases.sh` 中的别名加进你的 shell 配置，然后用 `source ~/.zshrc`（或 `~/.bashrc`）重新加载。
3. 用 `for i in $(seq 1 100); do echo "epoch $i loss: $(echo "scale=4; 1/$i" | bc)"; sleep 0.1; done > fake_train.log` 生成一个假训练日志，然后用 `grep`、`tail` 和 `awk` 提取出其中的损失值。
4. 为一台你有权限访问的服务器配置一条 SSH config 记录（或者用 `localhost` 练习语法）。

## 关键术语

| 术语 | 大家怎么叫 | 实际含义 |
|------|----------------|----------------------|
| Shell | “终端” | 解释执行你输入命令的程序（bash、zsh、fish） |
| tmux | “终端复用器” | 让你在一个窗口里运行多个终端会话，并支持断开/重连的程序 |
| Pipe | “那根竖线” | `\|` 操作符，把一个命令的输出作为另一个命令的输入 |
| PID | “进程 ID” | 分配给每个运行中进程的唯一编号，用于监控或杀掉进程 |
| nohup | “No hangup（不挂断）” | 让命令免疫挂断信号，关闭终端也不会杀死它 |
| SSH | “连服务器” | Secure Shell，一种在远程机器上执行命令的加密协议 |
