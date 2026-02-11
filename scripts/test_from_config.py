# 导入PyTorch库，作为深度学习的核心框架，用于张量运算、模型构建和自动求导等
import torch
# 导入时间模块，用于记录模型推理耗时
import time
# 从tensorly库导入张量代数模块，tensorly是专门用于张量运算的库
from tensorly import tenalg

# 设置tensorly的张量运算后端为"einsum"
# einsum（爱因斯坦求和约定）是一种灵活的张量收缩/运算方式，适合高维张量计算
tenalg.set_backend("einsum")

# 从neuralop库导入模型构建函数，neuralop是专注于神经算子（Neural Operator）的深度学习库
# 神经算子主要用于偏微分方程求解、流体力学等科学计算领域的算子学习
from neuralop import get_model


# -------------------------- 配置读取相关模块导入 --------------------------
# 从zencfg库导入命令行配置解析函数，zencfg用于便捷地管理和解析配置文件/命令行参数
from zencfg import make_config_from_cli
# 导入系统模块，用于修改Python的模块搜索路径
import sys
sys.path.insert(0, "d:\\document\\python\\neural operators")
print(sys.path)  # 打印当前的Python路径列表，验证上级目录是否成功添加到搜索路径
from config.test_config import TestConfig


# -------------------------- 配置解析与参数提取 --------------------------
# 从命令行参数解析配置，并创建TestConfig实例（自动解析CLI传入的参数并填充到配置类中）
config = make_config_from_cli(TestConfig)
# 打印完整的配置对象，查看所有配置项
print(config)
# 将配置对象转为字典格式，方便后续按key取值
print(config.to_dict())
# 赋值为字典格式的配置，简化后续参数读取
configdic = config.to_dict()

# 从配置中提取数据批次大小（每次前向传播的样本数量）
batch_size = config.data.batch_size
# # 从配置中提取输入数据的空间尺寸（如二维数据的宽/高）
# size = config.data.size

# -------------------------- 计算设备配置 --------------------------
# 判断当前环境是否支持CUDA（NVIDIA GPU加速）
if torch.has_cuda:
    # 若支持CUDA，使用GPU作为计算设备
    device = "cuda"
else:
    # 若不支持CUDA，使用CPU作为计算设备
    device = "cpu"

# -------------------------- 模型初始化 --------------------------
# 根据配置文件创建神经算子模型（模型结构、超参数等由config指定）
model = get_model(config)
# 将模型的所有参数和缓冲区移到指定设备（CPU/GPU）
model = model.to(device)

# -------------------------- 生成测试输入数据 --------------------------
# 生成随机输入张量：
# - batch_size：批次大小
# - 3：输入通道数（如RGB图像的3通道，或科学计算中的3个物理量维度）
# - size x size：输入的空间维度（二维张量的尺寸）
# - torch.randn：生成标准正态分布的随机数，模拟输入数据
# - .to(device)：将张量移到指定计算设备
in_data = torch.randn(batch_size, 3, size, size).to(device)

# 打印模型的类名（查看模型具体类型，如FNO、DeepONet等神经算子模型）
print(model.__class__)
# 打印模型的完整结构（包括各层的参数、维度等）
print(model)

# -------------------------- 模型前向推理耗时测试 --------------------------
# 记录前向推理开始时间
t1 = time.time()
# 执行模型前向传播：将输入数据传入模型，得到输出结果
out = model(in_data)
# 计算前向推理耗时（结束时间 - 开始时间）
t = time.time() - t1
# 打印输出张量的形状和推理耗时，验证模型是否正常输出
print(f"Output of size {out.shape} in {t}.")

# -------------------------- 反向传播与梯度检查 --------------------------
# 构造简单的损失函数：将输出张量所有元素求和（仅用于测试反向传播，无实际业务意义）
loss = out.sum()
# 执行反向传播：计算损失对所有可训练参数的梯度（自动求导）
loss.backward()

# -------------------------- 检查未使用的模型参数 --------------------------
# 遍历模型的所有命名参数（name：参数名，param：参数张量）
for name, param in model.named_parameters():
    # 若参数的梯度为None，说明该参数在本次前向传播中未被使用（未参与计算图）
    if param.grad is None:
        # 打印未使用的参数名，提示可能存在的模型设计问题
        print(f"Usused parameter {name}!")