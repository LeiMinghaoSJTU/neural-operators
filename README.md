本库展示了一系列神经算子(Neural Operator)的实现，用于高效求解偏微分方程(PDE)。

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