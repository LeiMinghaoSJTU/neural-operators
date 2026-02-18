本库展示了一系列神经算子(Neural Operator)的实现，用于高效求解偏微分方程(PDE)。Neuralop 库所支持的神经算子类型包括：

- FNO（傅里叶神经算子）：基于Fourier变换的神经算子（1D、2D、3D变体）

- TFNO（张量化FNO）：带有Tucker分解的张量化版本（1D、2D、3D变体）

- SFNO（球面FNO）：基于球谐波的球面谐波FNO（需要torch_harmonics）

- UNO（U形神经算子）：受U-Net启发的神经算子架构

- UQNO（不确定性量化NO）：具有不确定性量化的神经算子

- FNOGNO（FNO + 图神经运算符）：混合 FNO-GNO 架构

- GINO（几何信息神经算子）：用于不规则域的基于图的神经算子

- LocalNO：用于高效计算的局部神经算子（需要torch_harmonics）

- CODANO：连续离散神经算子


神经算子库的英文说明[点此查看](https://neuraloperator.github.io/dev/index.html)。

## 系统要求
- Python 3.8 - 3.12 (暂不支持 Python 3.13 +)
- CUDA 11.0+ (如需GPU加速)



## 安装依赖库

从清华镜像源安装依赖库：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 运行示例

为了运行 .ipynb 示例，您需要安装 Jupyter Notebook。请确保已安装 Jupyter Kernel：

```bash
pip install notebook
```


## 可视化训练过程

请修改 config 文件夹中的 wandb_api_key.txt 文件，将 WandB 的 API Key 填写进去。如果您尚未创建过 WandB 账号，请先[注册](https://wandb.ai)。
