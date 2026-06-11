# AI 工程中的 Linux

> 绝大多数 AI 都跑在 Linux 上。你需要掌握足够的知识，才不会被卡住。

**Type:** Learn
**Languages:** --
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~30 minutes

## 学习目标

- 在命令行中浏览 Linux 文件系统并完成基本的文件操作
- 使用 `chmod` 和 `chown` 管理文件权限，解决 "Permission denied" 报错
- 使用 `apt` 安装系统软件包，并把一台全新的 GPU 机器配置好用于 AI 工作
- 识别那些常让开发者在远程机器上踩坑的 macOS 与 Linux 差异

## 问题背景

你平时在 macOS 或 Windows 上开发。但只要你 SSH 进一台云端 GPU 机器、租一台 Lambda 实例，或启动一台 EC2 机器，你面对的就是 Ubuntu。终端是你唯一的界面。没有 Finder，没有资源管理器，没有图形界面。如果你不会在命令行里浏览文件系统、安装软件包、管理进程，你就只能一边为闲置的 GPU 时长付费，一边在网上搜"Linux 怎么解压文件"。

这是一份生存指南。它只涵盖你在远程 Linux 机器上做 AI 工作所必需的内容，绝不多讲。

## 文件系统布局

Linux 把所有东西都组织在唯一的根目录 `/` 之下。没有 `C:\`，也没有 `/Volumes`。你真正会用到的目录：

```mermaid
graph TD
    root["/"] --> home["home/your-username/<br/>Your files — clone repos, run training"]
    root --> tmp["tmp/<br/>Temporary files, cleared on reboot"]
    root --> usr["usr/<br/>System programs and libraries"]
    root --> etc["etc/<br/>Config files"]
    root --> varlog["var/log/<br/>Logs — check when something breaks"]
    root --> mnt["mnt/ or /media/<br/>External drives and volumes"]
    root --> proc["proc/ and /sys/<br/>Virtual files — kernel and hardware info"]
```

你的主目录是 `~`，也就是 `/home/your-username`。你做的几乎所有事情都发生在这里。

## 基本命令

下面这 15 个命令能覆盖你在远程 GPU 机器上 95% 的操作。

### 目录导航

```bash
pwd                         # Where am I?
ls                          # What's here?
ls -la                      # What's here, including hidden files with details?
cd /path/to/dir             # Go there
cd ~                        # Go home
cd ..                       # Go up one level
```

### 文件与目录

```bash
mkdir my-project            # Create a directory
mkdir -p a/b/c              # Create nested directories in one shot

cp file.txt backup.txt      # Copy a file
cp -r src/ src-backup/      # Copy a directory (recursive)

mv old.txt new.txt          # Rename a file
mv file.txt /tmp/           # Move a file

rm file.txt                 # Delete a file (no trash, it's gone)
rm -rf my-dir/              # Delete a directory and everything inside
```

`rm -rf` 是不可逆的。没有撤销。按回车之前务必再检查一遍路径。

### 查看文件内容

```bash
cat file.txt                # Print entire file
head -20 file.txt           # First 20 lines
tail -20 file.txt           # Last 20 lines
tail -f log.txt             # Follow a log file in real time (Ctrl+C to stop)
less file.txt               # Scroll through a file (q to quit)
```

### 搜索

```bash
grep "error" training.log           # Find lines containing "error"
grep -r "learning_rate" .           # Search all files in current directory
grep -i "cuda" config.yaml          # Case-insensitive search

find . -name "*.py"                 # Find all Python files under current dir
find . -name "*.ckpt" -size +1G     # Find checkpoint files larger than 1GB
```

## 权限

Linux 中的每个文件都有所有者和权限位。当脚本无法执行，或者你无法写入某个目录时，你就会撞上它。

```bash
ls -l train.py
# -rwxr-xr-- 1 user group 2048 Mar 19 10:00 train.py
#  ^^^             owner permissions: read, write, execute
#     ^^^          group permissions: read, execute
#        ^^        everyone else: read only
```

常见修复方法：

```bash
chmod +x train.sh           # Make a script executable
chmod 755 deploy.sh         # Owner: full, others: read+execute
chmod 644 config.yaml       # Owner: read+write, others: read only

chown user:group file.txt   # Change who owns a file (needs sudo)
```

凡是提示 "Permission denied" 的，几乎都是权限问题。`chmod +x` 或 `sudo` 能解决绝大多数情况。

## 包管理（apt）

Ubuntu 使用 `apt`。这是你安装系统级软件的方式。

```bash
sudo apt update             # Refresh the package list (always do this first)
sudo apt install -y htop    # Install a package (-y skips confirmation)
sudo apt install -y build-essential  # C compiler, make, etc. Needed by many Python packages
sudo apt install -y tmux    # Terminal multiplexer (keep sessions alive after disconnect)

apt list --installed        # What's installed?
sudo apt remove htop        # Uninstall
```

在一台全新的 GPU 机器上，你通常会安装这些常用软件包：

```bash
sudo apt update && sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    tmux \
    htop \
    unzip \
    python3-venv
```

## 用户与 sudo

你通常以普通用户身份登录。有些操作需要 root（管理员）权限。

```bash
whoami                      # What user am I?
sudo command                # Run a single command as root
sudo su                     # Become root (exit to go back, use sparingly)
```

在云端 GPU 实例上，你通常是唯一的用户，并且已经拥有 sudo 权限。不要什么都用 root 跑。只在需要时才用 sudo。

## 进程与 systemd

当训练卡住，或者你想看看机器上在跑什么时：

```bash
htop                        # Interactive process viewer (q to quit)
ps aux | grep python        # Find running Python processes
kill 12345                  # Gracefully stop process with PID 12345
kill -9 12345               # Force kill (use when graceful doesn't work)
nvidia-smi                  # GPU processes and memory usage
```

systemd 负责管理服务（后台守护进程）。如果你要运行推理服务器，就会用到它：

```bash
sudo systemctl start nginx          # Start a service
sudo systemctl stop nginx           # Stop it
sudo systemctl restart nginx        # Restart it
sudo systemctl status nginx         # Check if it's running
sudo systemctl enable nginx         # Start automatically on boot
```

## 磁盘空间

GPU 机器的磁盘空间往往有限。模型和数据集很快就会把它填满。

```bash
df -h                       # Disk usage for all mounted drives
df -h /home                 # Disk usage for /home specifically

du -sh *                    # Size of each item in current directory
du -sh ~/.cache             # Size of your cache (pip, huggingface models land here)
du -sh /data/checkpoints/   # Check how big your checkpoints are

# Find the biggest space hogs
du -h --max-depth=1 / 2>/dev/null | sort -hr | head -20
```

常用的腾空间手段：

```bash
# Clear pip cache
pip cache purge

# Clear apt cache
sudo apt clean

# Remove old checkpoints you don't need
rm -rf checkpoints/epoch_01/ checkpoints/epoch_02/
```

## 网络

你会在命令行里下载模型、传输文件、调用 API。

```bash
# Download files
wget https://example.com/model.bin                   # Download a file
curl -O https://example.com/data.tar.gz              # Same thing with curl
curl -s https://api.example.com/health | python3 -m json.tool  # Hit an API, pretty-print JSON

# Transfer files between machines
scp model.bin user@remote:/data/                     # Copy file to remote machine
scp user@remote:/data/results.csv .                  # Copy file from remote to local
scp -r user@remote:/data/checkpoints/ ./local-dir/   # Copy directory

# Sync directories (faster than scp for large transfers, resumes on failure)
rsync -avz --progress ./data/ user@remote:/data/
rsync -avz --progress user@remote:/results/ ./results/
```

传大文件时用 `rsync` 而不是 `scp`。它只传输发生变化的字节，而且能在连接中断后续传。

## tmux：让会话保持存活

SSH 连到远程机器后，一合上笔记本电脑，你的训练任务就会被杀掉。tmux 可以避免这种情况。

```bash
tmux new -s train           # Start a new session named "train"
# ... start your training, then:
# Ctrl+B, then D            # Detach (training keeps running)

tmux ls                     # List sessions
tmux attach -t train        # Reattach to session

# Inside tmux:
# Ctrl+B, then %            # Split pane vertically
# Ctrl+B, then "            # Split pane horizontally
# Ctrl+B, then arrow keys   # Switch between panes
```

长时间的训练任务永远放在 tmux 里跑。没有例外。

## Windows 用户的 WSL2

如果你用 Windows，WSL2 能让你获得一个真正的 Linux 环境，不需要装双系统。

```bash
# In PowerShell (admin)
wsl --install -d Ubuntu-24.04

# After restart, open Ubuntu from Start menu
sudo apt update && sudo apt upgrade -y
```

WSL2 运行的是真正的 Linux 内核。本课的所有内容都能在其中正常工作。在 WSL 内部，你的 Windows 文件位于 `/mnt/c/Users/YourName/`。

GPU 直通需要在 Windows 这一侧安装 NVIDIA 驱动。安装 Windows 版的 NVIDIA 驱动（而不是 Linux 版），CUDA 就能在 WSL2 内部使用。

## 常见陷阱：从 macOS 到 Linux

如果你从 macOS 转过来，这些地方会让你踩坑：

| macOS | Linux | 说明 |
|-------|-------|-------|
| `brew install` | `sudo apt install` | 包名有时不一样。`brew install htop` 和 `sudo apt install htop` 效果相同，但 `brew install readline` 和 `sudo apt install libreadline-dev` 就不是了。 |
| `open file.txt` | `xdg-open file.txt` | 不过远程机器上没有图形界面。用 `cat` 或 `less`。 |
| `pbcopy` / `pbpaste` | 不可用 | 通过管道读写剪贴板这件事在 SSH 上不存在。 |
| `~/.zshrc` | `~/.bashrc` | macOS 默认是 zsh。多数 Linux 服务器用 bash。 |
| `/opt/homebrew/` | `/usr/bin/`、`/usr/local/bin/` | 二进制文件存放的位置不同。 |
| `sed -i '' 's/a/b/' file` | `sed -i 's/a/b/' file` | macOS 的 sed 在 `-i` 后面需要一个空字符串，Linux 不需要。 |
| 文件系统不区分大小写 | 文件系统区分大小写 | 在 Linux 上，`Model.py` 和 `model.py` 是两个不同的文件。 |
| 行尾符 `\n` | 行尾符 `\n` | 两者相同。但 Windows 用 `\r\n`，会让 bash 脚本运行出错。用 `dos2unix` 修复。 |

## 快速参考卡

```
Navigation:     pwd, ls, cd, find
Files:          cp, mv, rm, mkdir, cat, head, tail, less
Search:         grep, find
Permissions:    chmod, chown, sudo
Packages:       apt update, apt install
Processes:      htop, ps, kill, nvidia-smi
Services:       systemctl start/stop/restart/status
Disk:           df -h, du -sh
Network:        curl, wget, scp, rsync
Sessions:       tmux new/attach/detach
```

## 练习

1. SSH 进任意一台 Linux 机器（或打开 WSL2），进入主目录。创建一个项目文件夹，用 `touch` 在里面创建三个空文件，再用 `ls -la` 列出它们。
2. 用 apt 安装 `htop`，运行它，并找出占用内存最多的进程。
3. 启动一个 tmux 会话，在里面运行 `sleep 300`，分离会话，列出所有会话，再重新连接。
4. 用 `df -h` 查看可用磁盘空间，再用 `du -sh ~/.cache/*` 找出缓存中占空间的内容。
5. 先用 `scp` 把一个文件从本地传到远程机器，再用 `rsync` 做同样的传输，对比两者的体验。
