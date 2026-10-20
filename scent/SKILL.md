---
name: scnet-guide
description: >-
  Standard operating procedures, best practices, troubleshooting, and optimization tricks
  for conducting deep learning experiments on the SCNet (National Supercomputing Internet /
  国家超算互联网) cluster using Slurm, PyTorch AMP, and Hygon C86 environments. Use when the user
  asks to run tasks, debug errors, configure SSH/Slurm, or optimize performance on SCNet.
---

# SCNet (国家超算互联网平台) 深度学习全流程实战指南

本指南记录了在 SCNet 超算集群上从**精准依赖管理**、**连接与环境调度**、**双流与 3D 模型高效训练**到**排障避坑**的标准化单文件操作准则。

---

## 1. 前置核心:规范化依赖管理与 `pyproject.toml` 标准模板

在 SCNet 上开展任何 Python 任务前,**绝对严禁直接运行 `uv pip install` 或裸 `pip install`**。临时命令式安装会导致依赖漂移、破坏锁文件、无法跨节点复现,并在集群多环境下极易引发依赖地狱。必须预先编写声明式的 `pyproject.toml`,并通过 `uv sync --extra cuda` 进行精确依赖锁定与安装。

### 1.1 超算底层约束与关键配置哲学
1. **较老系统与运行时环境**:
   - 登录节点与计算节点基于 **CentOS 7**(Linux 3.10 内核,**glibc 2.17**);
   - 驱动版本:550.135,底层 CUDA 版本:**12.4 (`cu124`)**;
   - 节点缺少高版本 GCC/Clang 编译器与开发头文件。
2. **强制使用预编译 Wheel(禁源码编译)**:
   - 在 `[tool.uv]` 中必须配置 `no-build = true`,强制禁止从源码编译(Build),所有依赖必须下载已适配 `manylinux2014`(glibc 2.17 兼容)的官方二进制预编译 Wheel,彻底规避因底层 C/C++ 依赖缺失造成的编译中断崩溃。
3. **全链路国内镜像加速**:
   - 默认 PyPI 索引采用**清华源 (Tsinghua)**;
   - PyTorch CUDA 12.4 专用包采用**南京大学 (NJU) 镜像源**,杜绝海外下载限速与连接重置超时;
   - 保留官方 CPU 索引作为安全回退。
4. **依赖安装必须在[登录节点]进行**:
   - 计算节点处于物理隔离内网,完全无法联网;
   - 登录节点外网可通,依赖同步一次后会自动落盘在共享存储 `/work/home/<user>`,计算节点直接激活即可。

### 1.2 生产级 `pyproject.toml` 标准模板
在项目根目录创建或维护该配置文件:

```toml
[project]
name = "scnet-dl-project"
version = "0.1.0"
description = "Deep learning project optimized for SCNet cluster"
readme = "README.md"
requires-python = ">=3.11,<3.12"

# 基础运行时依赖(必须锁定精准兼容版本)
dependencies = [
    "numpy==1.26.4",
    "scipy==1.13.1",
    "scikit-learn==1.5.2",
    "polars==1.8.2",
    "pillow>=10.0.0",
    "tqdm>=4.66.5",
    "pytorchvideo>=0.1.5",
]

# 严格区分 CPU 与 CUDA 12.4 依赖分支
[project.optional-dependencies]
cpu = ["torch==2.6.0", "torchvision==0.21.0"]
cuda = ["torch==2.6.0", "torchvision==0.21.0"]

# 1. 国内默认主索引:清华源
[[tool.uv.index]]
name = "tsinghua"
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true

# 2. PyTorch 官方 CPU 回退源
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

# 3. 南京大学 PyTorch CUDA 12.4 专用高速镜像
[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://mirrors.nju.edu.cn/pytorch/whl/cu124"
explicit = true

# 依赖来源精确映射
[tool.uv.sources]
torch = [
    { index = "pytorch-cpu", extra = "cpu" },
    { index = "pytorch-cuda", extra = "cuda" },
]
torchvision = [
    { index = "pytorch-cpu", extra = "cpu" },
    { index = "pytorch-cuda", extra = "cuda" },
]

[tool.uv]
# 互斥分支声明与绝对禁止源码编译
conflicts = [[{ extra = "cpu" }, { extra = "cuda" }]]
no-build = true
```

### 1.3 登录节点依赖一键同步命令
在登录节点(如 `login04`)的项目目录下执行:
```bash
# 精准安装 CUDA 12.4 预编译依赖并生成 uv.lock
uv sync --extra cuda
```

---
## 2. 必备"黄金三法则"环境变量

计算节点启动任何 Python 任务前,**必须在 Shell 脚本最前置导出以下三个环境变量**:

```bash
export HF_HUB_OFFLINE=1 POLARS_SKIP_CPU_CHECK=1 PYTHONUNBUFFERED=1
```

| 环境变量 | 根治问题与底层机理 | 未配置的严重后果 |
| :--- | :--- | :--- |
| `POLARS_SKIP_CPU_CHECK=1` | 国产海光 CPU (Hygon C86) 缺少 Intel 专有 `pclmulqdq` 指令集 | 导入 `polars` 直接引发 `RuntimeWarning` 或非法指令错误中断非交互进程 |
| `HF_HUB_OFFLINE=1` | 计算节点处于物理隔离内网,无公网连接权限 | HuggingFace 自动请求远端元数据,网络超时死锁崩溃 |
| `PYTHONUNBUFFERED=1` | 批处理或后台运行下 Python 默认启用行/块级缓冲区 | `print` 与进度条无法实时写出到日志文件,导致监控黑盒,直到进程终止才刷盘 |

> **数据离线同步法则**:若需要 HuggingFace 模型或数据集,先在有外网的机器下载到本地缓存 `~/.cache/huggingface/`,再通过 `rsync -avz ~/.cache/huggingface/ scnet:~/.cache/huggingface/` 同步至超算共享家目录。

---

## 3. Slurm 调度与任务执行规范

### 3.1 生产级批处理提交 `sbatch`(推荐)
通过脚本提交后台作业,不受本地 SSH 掉线或电脑休眠影响,任务结束后**自动释放节点停止扣费**。

#### 标准 `run_train.sh` 脚本模板:
```bash
#!/bin/bash
#SBATCH -p xhhgnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:1
#SBATCH -t 01:30:00
#SBATCH -J dl_train
#SBATCH -o /work/home/guosichen/my-project/train.log
#SBATCH -e /work/home/guosichen/my-project/train.log

cd /work/home/guosichen/my-project
source .venv/bin/activate

# 导出黄金三法则环境变量
export HF_HUB_OFFLINE=1 POLARS_SKIP_CPU_CHECK=1 PYTHONUNBUFFERED=1

echo "=========================================================="
echo "  [SCNet 任务启动]  时间: $(date)"
echo "  执行节点: $(hostname)"
echo "  GPU 状态:"
nvidia-smi
echo "=========================================================="

# 必须加 -u 保证日志实时刷新
python -u train.py --epochs 15 --batch-size 16 --lr 1e-4

echo "=========================================================="
echo "  [SCNet 任务完成]  时间: $(date)"
echo "=========================================================="
```

#### 提交与监控命令:
```bash
# 提交作业
sbatch run_train.sh

# 查看当前运行中的作业
squeue -u $USER

# 实时查看输出日志
tail -f train.log
```

### 3.2 交互式调试 `salloc` 流程与资源回收警示
用于断点调试或代码冒烟测试。**测试完成必须立刻释放**,否则将持续耗费机时配额!

```bash
# 1. 申请 30 分钟计算节点资源 (3080/3080Ti, 16 CPU)
salloc -p xhhgnormal -N 1 --ntasks-per-node=16 --gres=gpu:1 -t 00:30:00

# 2. 查看分配到的节点主机名 (如 e03r04)
squeue -u $USER

# 3. 手动 SSH 进入该计算节点
ssh e03r04

# 4. 进入项目目录并激活环境
cd /work/home/guosichen/my-project
source .venv/bin/activate
export HF_HUB_OFFLINE=1 POLARS_SKIP_CPU_CHECK=1 PYTHONUNBUFFERED=1
python -u train.py --epochs 1

# 5. 调试结束:退出节点并立即释放资源 (必须执行)
exit
scancel <JOBID>
```

### 3.3 执行边界铁律:可视化与文档生成任务禁止远端执行

**规则**:任何涉及**绘图、生成图片/PDF/SVG、LaTeX 编译、浏览器渲染(HTML→PDF、页面截图)**的任务,**一律不得在登录节点或计算节点执行**。

| 角色 | 职责边界 |
| :--- | :--- |
| **远端(登录/计算节点)** | 只做计算,只产出**结构化数据**:`metrics.jsonl`、`results.csv`、`preds.npz`、`*.parquet` |
| **本地(个人机器)** | 通过 `rsync` 拉取上述数据后,在**本地临时环境**执行绘图与文档生成 |
也不要在 `pyproject.toml` 中新增画图等依赖,直接uv --with 临时执行即可

避免每次繁琐地"先连登录节点、再查节点、再跳登录",在本地 `~/.ssh/config` 中配置通配跳转:

```ssh-config
Host scnet
    HostName xh5.hpccube.com
    User guosichen
    Port 65061
    IdentityFile ~/.ssh/guosichen_scnet.pem
    GSSAPIAuthentication no
    ControlMaster auto
    ControlPath ~/.ssh/controlmasters/%C
    ControlPersist 10m

# 通配所有 e 开头的计算节点 (e01r*, e03r*, e07r*, e11r* 等)
Host e*
    User guosichen
    IdentityFile ~/.ssh/guosichen_scnet.pem
    StrictHostKeyChecking no
    ProxyCommand ssh -W %h:%p scnet
    GSSAPIAuthentication no
    Compression yes
    ControlMaster auto
    ControlPath ~/.ssh/controlmasters/%C
    ControlPersist 10m
```

**配置后的效率飞跃**:
- 本地终端直接登录已分配节点:`ssh e03r05`
- 本地文件直接传送至计算节点:`scp train.py e03r05:/work/home/guosichen/...`
- 本地直接向计算节点增量同步:`rsync -avz dataset/ e03r05:/work/home/guosichen/dataset/`

---

## 5. 排障避坑与错误速查表

| 错误现象 | 根本诱因 | 解决方案 |
| :--- | :--- | :--- |
| **`/bin/bash: squeue: 未找到命令`** | 远程 SSH 命令执行未加载登录环境,缺 `/opt/gridview/slurm/bin/` | 必须使用 `ssh scnet "bash -l -c '...'"` 执行命令 |
| **`squeue` 持续显示 `PD (Priority)`** | 该分区所有节点 GPU 均被占用,处于 Backfill 调度等待队列中 | 查看预计排期:`scontrol show job <JOBID>`;轻量评估作业可改小申请时间(如 10 分钟)进行 Backfill 快速插队 |
| **`Job exceeded its time limit`** | 作业申请时长不足,被调度器强制撤销 | 预估训练时间并在申请时留足 30% 缓冲;务必实施上述"增量持久化准则" |
| **`Polars RuntimeWarning / SIGILL`** | 海光 Hygon CPU 缺失 Intel `pclmulqdq` 指令集 | 启动前配置 `export POLARS_SKIP_CPU_CHECK=1` |
