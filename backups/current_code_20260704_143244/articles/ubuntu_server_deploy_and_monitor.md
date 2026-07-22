# Ubuntu 服务器部署与运行监控指南

---

## 目录

1. [环境准备](#1-环境准备)
2. [项目上传](#2-项目上传)
3. [命令对照：Windows → Ubuntu](#3-命令对照windows--ubuntu)
4. [后台运行与监控](#4-后台运行与监控)
5. [批量实验脚本](#5-批量实验脚本)
6. [结果回传](#6-结果回传)
7. [常见问题](#7-常见问题)

---

## 1. 环境准备

### 1.1 安装 Python + 虚拟环境

```bash
# Ubuntu 22.04/24.04 自带 Python 3.10+，确认版本
python3 --version        # 应 ≥ 3.10

# 创建虚拟环境
python3 -m venv ~/venvs/usv_swarm

# 激活
source ~/venvs/usv_swarm/bin/activate

# 升级 pip
pip install --upgrade pip setuptools wheel
```

### 1.2 安装依赖

```bash
# 核心依赖（pyproject.toml 中的依赖）
pip install 'casadi>=3.7' 'matplotlib>=3.8' 'numpy>=2.0' 'pillow>=10.0'
```

> **CasADi 注意**：如果 `pip install casadi` 失败（极少数情况），可以试试：
> ```bash
> pip install casadi --no-build-isolation
> # 或从 conda 安装
> conda install -c conda-forge casadi
> ```

### 1.3 验证安装

```bash
python3 -c "
import casadi; print('CasADi:', casadi.__version__)
import numpy;  print('NumPy:', numpy.__version__)
import matplotlib; print('Matplotlib:', matplotlib.__version__)
print('OK')
"
```

---

## 2. 项目上传

```bash
# 方式一：scp（从 Windows 上传）
# 在 Windows PowerShell 中运行：
scp -r d:\code\work1_update\ user@your-server:/home/user/usv_swarm/

# 方式二：rsync（推荐，支持断点续传）
rsync -avz --progress \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'outputs/' \
  --exclude 'backups/' \
  d:/code/work1_update/ \
  user@your-server:/home/user/usv_swarm/

# 方式三：git（如果已初始化仓库）
cd d:\code\work1_update
git init && git add -A && git commit -m "init"
git remote add origin user@your-server:/home/user/usv_swarm.git
git push origin main
```

上传后在服务器上确认结构：

```bash
cd ~/usv_swarm
ls -la
# 应看到：src/ examples/ maps/ outputs/ articles/ tests/
```

---

## 3. 命令对照：Windows → Ubuntu

### 3.1 前缀变化

| Windows | Ubuntu |
|---|---|
| `D:\anaconda3\envs\pytorch_gpu\python.exe` | `python3`（虚拟环境激活后） |
| `\` (反斜杠) | `/` (正斜杠) |
| `maps\...\xxx.json` | `maps/.../xxx.json` |

### 3.2 完整命令示例

```bash
# 切换到项目目录
cd ~/usv_swarm

# ★ 最常用：15×15 + FA³ACO
python3 examples/run_paper_style_region_tsp_experiment.py \
  --map maps/static_obstacle_map_15x15_rect_triangle_small/static_obstacle_map_15x15_rect_triangle_small.json \
  --rmin 0.5 \
  --tsp-solver fa3aco \
  --aco-ants 40 \
  --aco-iterations 120 \
  --aco-seed 42 \
  --dpi 120

# 20×20 + ACO
python3 examples/run_paper_style_region_tsp_experiment.py \
  --map maps/static_obstacle_map_20x20_two_obstacles/static_obstacle_map_20x20_two_obstacles.json \
  --rmin 0.8 \
  --tsp-solver aco \
  --aco-ants 50 \
  --aco-iterations 150 \
  --aco-seed 42 \
  --dpi 120

# 50×50 大地图 + FA³ACO
python3 examples/run_paper_style_region_tsp_experiment.py \
  --map maps/static_obstacle_map_50x50_simple/static_obstacle_map_50x50_simple.json \
  --rmin 1.0 \
  --tsp-solver fa3aco \
  --aco-ants 40 \
  --aco-iterations 100 \
  --aco-seed 1234 \
  --dpi 120

# 路径规划层 Demo
python3 examples/path_planning_layer_demo.py \
  --agents 4 \
  --tsp-solver fa3aco \
  --aco-ants 40 \
  --aco-iterations 120 \
  --aco-seed 42

# 闭环仿真
python3 examples/closed_loop_simulation.py \
  --total-time 45.0 \
  --fps 10 \
  --output outputs/closed_loop_45s.gif
```

> **提示**：Ubuntu 上无需写完整 Python 路径，`python3` 就是虚拟环境中的 Python。

---

## 4. 后台运行与监控

大型实验（如 50×50 大地图 + ACO × 120 迭代）可能运行 5~30 分钟。以下是生产级后台运行方案。

### 4.1 方案对比

| 方案 | 断连后存活 | 可回看输出 | 资源监控 | 推荐场景 |
|---|---|---|---|---|
| `nohup` | ✓ | 日志文件 | ✗ | 单次实验 |
| `screen` | ✓ | `screen -r` 回连 | ✗ | 交互式调试 |
| `tmux` | ✓ | `tmux attach` 回连 | ✗ | 交互式调试（推荐） |
| `nohup` + `htop` | ✓ | 日志文件 | ✓ | 批量实验（推荐） |

### 4.2 方案A：tmux（交互式，最推荐）

```bash
# 安装
sudo apt install tmux

# 创建新会话
tmux new -s usv_exp

# 在 tmux 中运行实验
cd ~/usv_swarm
source ~/venvs/usv_swarm/bin/activate
python3 examples/run_paper_style_region_tsp_experiment.py \
  --map maps/static_obstacle_map_50x50_simple/...json \
  --rmin 1.0 --tsp-solver fa3aco --aco-ants 40 --aco-iterations 100 --dpi 120

# 断开会话（程序继续运行）
# 按键: Ctrl+B, 然后按 D

# 重新连接
tmux attach -t usv_exp

# 查看所有会话
tmux ls

# 终止会话
tmux kill-session -t usv_exp
```

### 4.3 方案B：nohup + 日志文件（适合批量）

```bash
# 单次实验
nohup python3 examples/run_paper_style_region_tsp_experiment.py \
  --map maps/static_obstacle_map_50x50_simple/...json \
  --rmin 1.0 --tsp-solver fa3aco --aco-ants 40 --aco-iterations 100 --dpi 120 \
  > logs/exp_50x50_fa3aco.log 2>&1 &

# 记录 PID
echo $! > logs/exp_50x50_fa3aco.pid

# 查看实时输出
tail -f logs/exp_50x50_fa3aco.log
```

### 4.4 方案C：nohup + 资源监控脚本

创建监控脚本 `scripts/monitor.sh`：

```bash
#!/bin/bash
# 用法: bash scripts/monitor.sh <PID> <日志文件>

PID=$1
LOG=$2
INTERVAL=${3:-30}  # 默认30秒采样

echo "监控 PID=$PID, 日志=$LOG, 间隔=${INTERVAL}s"
echo "时间 | CPU% | MEM% | RSS(MB) | 日志末行"
echo "------|------|------|---------|--------"

while kill -0 $PID 2>/dev/null; do
    STATS=$(ps -p $PID -o %cpu,%mem,rss --no-headers 2>/dev/null || echo "0 0 0")
    CPU=$(echo $STATS | awk '{printf "%.1f", $1}')
    MEM=$(echo $STATS | awk '{printf "%.1f", $2}')
    RSS=$(echo $STATS | awk '{printf "%.0f", $3/1024}')
    LAST_LINE=$(tail -1 "$LOG" 2>/dev/null | cut -c1-60)
    echo "$(date +%H:%M:%S) | ${CPU}% | ${MEM}% | ${RSS}MB | ${LAST_LINE}"
    sleep $INTERVAL
done

echo "[$(date +%H:%M:%S)] 进程 $PID 已结束"
echo "完整输出见: $LOG"
```

使用方法：

```bash
mkdir -p logs

# 启动实验
nohup python3 examples/run_paper_style_region_tsp_experiment.py \
  --map maps/static_obstacle_map_50x50_simple/...json \
  --rmin 1.0 --tsp-solver fa3aco --aco-ants 40 --aco-iterations 100 --dpi 120 \
  > logs/exp_50x50_fa3aco.log 2>&1 &
PID=$!
echo $PID > logs/exp_50x50_fa3aco.pid

# 启动监控
bash scripts/monitor.sh $PID logs/exp_50x50_fa3aco.log 30
```

监控输出示例：

```
时间      | CPU%  | MEM% | RSS(MB) | 日志末行
----------|-------|------|---------|--------
14:30:05  | 98.5% | 2.3% | 458MB   | [ACO iter 45/100] best_obj=234.56
14:30:35  | 97.8% | 2.4% | 462MB   | [ACO iter 67/100] best_obj=231.12
14:31:05  | 99.1% | 2.4% | 465MB   | coverage_fraction: 0.978432
14:31:35  | 12.3% | 1.8% | 420MB   | paper_style_dir: outputs/...
[14:31:38] 进程 28473 已结束
```

### 4.5 实时查看实验状态

```bash
# 查看所有运行中的 Python 进程
ps aux | grep python3 | grep -v grep

# 查看各实验日志最后一行（判断当前阶段）
for f in logs/*.log; do
    echo "=== $(basename $f) ==="
    tail -1 "$f"
    echo
done

# 用 htop 看资源使用（按 CPU 排序）
htop -p $(pgrep -d',' python3)

# 用 nvidia-smi 看 GPU 使用（如果有 GPU 的 CasADi 版本）
watch -n 2 nvidia-smi
```

---

## 5. 批量实验脚本

### 5.1 完整批量实验脚本

创建 `scripts/batch_experiments.sh`：

```bash
#!/bin/bash
set -euo pipefail

source ~/venvs/usv_swarm/bin/activate
cd ~/usv_swarm

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BATCH_DIR="logs/batch_${TIMESTAMP}"
mkdir -p "$BATCH_DIR"

# 定义实验矩阵
# 格式: "地图路径 rmin USV数 求解器 蚂蚁数 迭代数 种子"
EXPERIMENTS=(
    # 15x15 - 3种求解器对比
    "maps/static_obstacle_map_15x15_rect_triangle_small/static_obstacle_map_15x15_rect_triangle_small.json 0.5 2 deterministic 0 0 0"
    "maps/static_obstacle_map_15x15_rect_triangle_small/static_obstacle_map_15x15_rect_triangle_small.json 0.5 2 aco 40 120 42"
    "maps/static_obstacle_map_15x15_rect_triangle_small/static_obstacle_map_15x15_rect_triangle_small.json 0.5 2 fa3aco 40 120 42"

    # 20x20 - 3种求解器对比
    "maps/static_obstacle_map_20x20_two_obstacles/static_obstacle_map_20x20_two_obstacles.json 0.8 2 deterministic 0 0 0"
    "maps/static_obstacle_map_20x20_two_obstacles/static_obstacle_map_20x20_two_obstacles.json 0.8 2 aco 50 150 42"
    "maps/static_obstacle_map_20x20_two_obstacles/static_obstacle_map_20x20_two_obstacles.json 0.8 2 fa3aco 50 150 42"

    # 50x50 - 3种求解器对比
    "maps/static_obstacle_map_50x50_simple/static_obstacle_map_50x50_simple.json 1.0 3 deterministic 0 0 0"
    "maps/static_obstacle_map_50x50_simple/static_obstacle_map_50x50_simple.json 1.0 3 aco 40 100 1234"
    "maps/static_obstacle_map_50x50_simple/static_obstacle_map_50x50_simple.json 1.0 3 fa3aco 40 100 1234"
)

echo "=========================================="
echo "批量实验开始: $(date)"
echo "实验总数: ${#EXPERIMENTS[@]}"
echo "批次目录: $BATCH_DIR"
echo "=========================================="

SUMMARY_FILE="$BATCH_DIR/summary.csv"
echo "地图,求解器,USV数,覆盖率,有效求解器,求解器状态,无效段数,碰撞段数,耗时(s)" > "$SUMMARY_FILE"

for i in "${!EXPERIMENTS[@]}"; do
    read -r MAP RMIN USV SOLVER ANTS ITERS SEED <<< "${EXPERIMENTS[$i]}"
    MAP_NAME=$(basename "$(dirname "$MAP")")
    EXP_NAME="${MAP_NAME}_${SOLVER}_usv${USV}"
    LOG_FILE="$BATCH_DIR/${EXP_NAME}.log"

    echo ""
    echo "--- [$(($i+1))/${#EXPERIMENTS[@]}] $EXP_NAME ---"
    echo "开始时间: $(date)"

    START_TIME=$(date +%s)

    # 构建命令
    if [ "$SOLVER" = "deterministic" ]; then
        CMD="python3 examples/run_paper_style_region_tsp_experiment.py \
            --map $MAP --rmin $RMIN --usv-count $USV \
            --tsp-solver deterministic --dpi 120 --no-render"
    else
        CMD="python3 examples/run_paper_style_region_tsp_experiment.py \
            --map $MAP --rmin $RMIN --usv-count $USV \
            --tsp-solver $SOLVER \
            --aco-ants $ANTS --aco-iterations $ITERS --aco-seed $SEED \
            --dpi 120"
    fi

    # 运行并捕获输出
    set +e
    nohup $CMD > "$LOG_FILE" 2>&1
    EXIT_CODE=$?
    set -e

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    # 解析结果
    COV=$(grep "coverage_fraction:" "$LOG_FILE" | tail -1 | awk '{print $NF}' || echo "N/A")
    EFF_SOLVER=$(grep "effective_tsp_solver:" "$LOG_FILE" | tail -1 | awk '{print $NF}' || echo "N/A")
    SOLVER_STATUS=$(grep "tsp_solver_status:" "$LOG_FILE" | tail -1 | awk '{print $NF}' || echo "N/A")
    INVALID=$(grep "invalid_path_length:" "$LOG_FILE" | tail -1 | awk '{print $NF}' || echo "N/A")
    COLLISION=$(grep "obstacle_collision_segment_count:" "$LOG_FILE" | tail -1 | awk '{print $NF}' || echo "N/A")

    echo "$MAP_NAME,$SOLVER,$USV,$COV,$EFF_SOLVER,$SOLVER_STATUS,$INVALID,$COLLISION,${ELAPSED}s" >> "$SUMMARY_FILE"

    echo "退出码=$EXIT_CODE 耗时=${ELAPSED}s 覆盖率=$COV 有效求解器=$EFF_SOLVER"
    echo "日志: $LOG_FILE"
done

echo ""
echo "=========================================="
echo "批量实验完成: $(date)"
echo "汇总文件: $SUMMARY_FILE"
echo "=========================================="
cat "$SUMMARY_FILE"
```

使脚本可执行并运行：

```bash
chmod +x scripts/batch_experiments.sh
nohup bash scripts/batch_experiments.sh > logs/batch_master.log 2>&1 &
echo $! > logs/batch_master.pid

# 监控批量进度
tail -f logs/batch_master.log

# 或在 tmux 中运行（推荐，可随时查看）
tmux new -s batch
bash scripts/batch_experiments.sh
# Ctrl+B, D 断开
```

### 5.2 快速对比脚本

创建 `scripts/quick_compare.sh`：

```bash
#!/bin/bash
# 在同一张地图上快速对比 3 种求解器
MAP=${1:-"maps/static_obstacle_map_15x15_rect_triangle_small/static_obstacle_map_15x15_rect_triangle_small.json"}
RMIN=${2:-0.5}

source ~/venvs/usv_swarm/bin/activate
cd ~/usv_swarm

for SOLVER in deterministic aco fa3aco; do
    echo "========== $SOLVER =========="
    if [ "$SOLVER" = "deterministic" ]; then
        python3 examples/run_paper_style_region_tsp_experiment.py \
            --map "$MAP" --rmin "$RMIN" --tsp-solver "$SOLVER" --dpi 120 --no-render 2>&1 | \
            grep -E "coverage_fraction|effective_tsp_solver|tsp_solver_status|invalid_path_length|planning_time"
    else
        python3 examples/run_paper_style_region_tsp_experiment.py \
            --map "$MAP" --rmin "$RMIN" --tsp-solver "$SOLVER" \
            --aco-ants 40 --aco-iterations 120 --aco-seed 42 --dpi 120 --no-render 2>&1 | \
            grep -E "coverage_fraction|effective_tsp_solver|tsp_solver_status|invalid_path_length|planning_time"
    fi
    echo
done
```

---

## 6. 结果回传

### 6.1 从服务器下载结果

```bash
# 在 Windows PowerShell 中运行：

# 下载指定地图的全部输出
scp -r user@your-server:/home/user/usv_swarm/outputs/static_obstacle_map_50x50_simple_* d:\code\work1_update\outputs\

# 下载所有输出
scp -r user@your-server:/home/user/usv_swarm/outputs/ d:\code\work1_update\outputs\

# 下载日志
scp user@your-server:/home/user/usv_swarm/logs/batch_*/summary.csv d:\code\work1_update\logs\
```

### 6.2 rsync 增量同步

```bash
# 在 Windows PowerShell 中（需要安装 rsync 或 WSL）：
rsync -avz --progress \
  user@your-server:/home/user/usv_swarm/outputs/ \
  d:/code/work1_update/outputs/

rsync -avz --progress \
  user@your-server:/home/user/usv_swarm/logs/ \
  d:/code/work1_update/logs/
```

### 6.3 在服务器上直接查看图片

如果不想下载，可以在服务器上安装轻量 HTTP 服务：

```bash
# 在 outputs 目录下启动临时 HTTP 服务
cd ~/usv_swarm/outputs
python3 -m http.server 8080

# 在浏览器中访问 http://your-server:8080/
# 浏览下载 PNG/GIF
```

---

## 7. 常见问题

### 7.1 matplotlib 报错 "no display"

```bash
# 确认 Agg 后端已设置（代码中已设置 matplotlib.use("Agg")，无需额外操作）
# 如果仍有问题：
export MPLBACKEND=Agg
python3 examples/...
```

### 7.2 CasADi 找不到求解器

```bash
# 检查是否安装了 IPOPT
python3 -c "import casadi; print(casadi.Opti().solver('ipopt'))"

# 如果失败，安装 IPOPT
sudo apt install coinor-libipopt-dev
# 或
conda install -c conda-forge ipopt
```

### 7.3 内存不足（50×50 地图 + ACO 可能用 2~4GB RAM）

```bash
# 限制 ACO 搜索规模
python3 examples/run_paper_style_region_tsp_experiment.py \
  --map maps/.../50x50.json \
  --tsp-solver fa3aco \
  --aco-ants 20 \           # 减少蚂蚁数
  --aco-iterations 50 \     # 减少迭代
  --dpi 80                  # 降低图片分辨率

# 或直接用 deterministic（内存友好）
python3 examples/run_paper_style_region_tsp_experiment.py \
  --map maps/.../50x50.json \
  --tsp-solver deterministic \
  --tsp-2opt-iterations 2
```

### 7.4 进程被 OOM Killer 杀死

```bash
# 查看系统日志
dmesg | grep -i kill | tail -10

# 增加 swap（临时）
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 用 cgroups 限制进程内存
systemd-run --scope -p MemoryMax=6G \
  python3 examples/run_paper_style_region_tsp_experiment.py ...
```

### 7.5 中止正在运行的实验

```bash
# 方式一：通过 PID 终止
kill $(cat logs/exp_50x50_fa3aco.pid)

# 方式二：通过进程名
pkill -f "run_paper_style_region_tsp_experiment"

# 方式三：强制终止（SIGKILL）
kill -9 $(pgrep -f "run_paper_style_region_tsp")

# 清理 tmux 会话
tmux kill-session -t usv_exp
```

---

## 快速启动检查清单

```bash
# 1. 连接服务器
ssh user@your-server

# 2. 激活环境
source ~/venvs/usv_swarm/bin/activate
cd ~/usv_swarm

# 3. 快速验证
python3 -c "from usv_swarm import plan_global_coverage; print('OK')"

# 4. 小规模试跑（< 30 秒）
python3 examples/path_planning_layer_demo.py --agents 2 --tsp-solver deterministic

# 5. 启动正式实验
tmux new -s main_exp
python3 examples/run_paper_style_region_tsp_experiment.py \
  --map maps/static_obstacle_map_15x15_rect_triangle_small/...json \
  --rmin 0.5 --tsp-solver fa3aco --aco-ants 40 --aco-iterations 120 --aco-seed 42 --dpi 120

# 6. 断开（Ctrl+B, D），之后随时回连
tmux attach -t main_exp
```
