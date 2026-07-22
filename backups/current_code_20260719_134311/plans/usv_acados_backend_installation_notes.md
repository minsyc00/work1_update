# Acados 后端说明与安装建议

## Acados 与原 CasADi/IPOPT 版本的区别

Acados 和原来的版本核心区别是：原来是 `CasADi + IPOPT` 通用非线性优化器；Acados 是面向实时 NMPC 的代码生成和快速 OCP 求解框架。

| 对比项 | CasADi/IPOPT | Acados |
|---|---|---|
| 定位 | 通用非线性规划求解 | 实时最优控制 / NMPC |
| 求解方式 | Python 调 CasADi Opti/IPOPT | 生成 C 求解器，再由 Python 调用 |
| 速度 | 通常几十到几百 ms/agent | 目标是 ms 级，更适合实时控制 |
| 实时性 | 多艇时容易卡顿 | 更适合高频闭环控制 |
| 安装难度 | 简单，`pip install casadi` | 复杂，需要编译 acados C 库 |
| 工程风险 | 稳定、容易调试 | 环境配置麻烦，Windows 原生更麻烦 |
| 当前项目策略 | 保留作为 fallback | 作为可选加速后端 |

可以把 CasADi/IPOPT 理解为“研究验证版求解器”，把 Acados 理解为“实时部署版求解器”。但 Acados 不是直接 `pip install acados` 就能使用，它需要先编译 C/C++ 库，再安装 Python 接口 `acados_template`。

## 推荐安装方式

当前主机是 Windows，推荐使用 `WSL/Ubuntu` 安装 Acados。官方文档也更推荐 Linux/Mac 或 Windows WSL 路线。

官方文档：

- Acados Installation: <https://docs.acados.org/installation/index.html>
- Python Interface Installation: <https://docs.acados.org/python_interface/index.html>

### WSL/Ubuntu 安装示例

```bash
sudo apt-get update
sudo apt-get install -y git make cmake build-essential python3-pip python3-virtualenv

git clone https://github.com/acados/acados.git
cd acados
git submodule update --recursive --init

mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j4
```

安装 Python 接口：

```bash
cd /path/to/acados
python3 -m virtualenv env
source env/bin/activate

pip install -e interfaces/acados_template
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/path/to/acados/lib"
export ACADOS_SOURCE_DIR="/path/to/acados"
```

官方特别提示：Python 接口推荐使用 `virtualenv`。`conda/miniconda` 可能有路径问题。

## 当前 Anaconda 环境注意事项

当前本地 Python 环境是：

```text
D:\anaconda3\envs\pytorch_gpu
```

如果一定要在该环境中直接使用 Acados，可能需要额外处理：

- 动态库路径。
- `ACADOS_SOURCE_DIR`。
- Windows 与 WSL 路径映射。
- `acados_template` 的 Python 包路径。

更稳妥的做法是：

1. 保留当前 Anaconda 环境运行项目主体和 CasADi fallback。
2. 单独建立 WSL Ubuntu 环境安装 Acados。
3. 先确认 Acados 官方 examples 能跑通。
4. 再让当前项目的 `auto/acados` 后端接入 Acados 环境。

## 当前项目中的使用方式

当前项目采用可选后端策略：

```bash
--nmpc-solver-backend auto
```

含义：优先尝试 Acados；如果环境没有 Acados，就自动回退 CasADi。

强制使用 CasADi：

```bash
--nmpc-solver-backend casadi
```

强制使用 Acados：

```bash
--nmpc-solver-backend acados
```

如果 Acados 没安装，强制模式会明确报错，不会假装成功。

## 建议

短期建议继续用 `auto` 或 `casadi` 保持项目稳定运行：

```bash
python examples/closed_loop_simulation.py --nmpc-solver-backend auto
```

等 Acados 环境安装并验证通过后，再用：

```bash
python examples/closed_loop_simulation.py --nmpc-solver-backend acados
```

这样既不会破坏当前 CasADi/IPOPT 版本，也能为后续实时 NMPC 性能优化保留清晰升级路径。
