'''
配置文件核心作用：这是一份针对Burgers 方程（流体力学经典方程） 的深度学习训练配置文件，基于zencfg配置框架管理所有训练相关参数，覆盖数据集、模型、优化器、分布式训练、实验追踪等全流程。
配置类设计逻辑：采用 “基类 + 子类” 的分层设计，ConfigBase为所有配置的基类，各子配置类（如BurgersDatasetConfig、BurgersOptConfig）负责单一模块，Default类整合所有子配置，便于统一管理和调用。
关键配置模块：核心模块包括数据集参数（分辨率、批次大小、样本数量）、优化参数（损失函数、学习率、调度器）、模型架构（FNO 傅里叶神经算子）、实验追踪（Wandb），所有参数均提供合理默认值，可根据任务需求灵活调整。
'''

# 导入Python类型提示模块，用于类型注解，提升代码可读性和类型检查能力
from typing import Any, List, Optional

# 导入配置基类ConfigBase，所有自定义配置类都继承自该类，提供配置管理的基础功能
from zencfg import ConfigBase
# 导入分布式训练相关的配置类
from .distributed import DistributedConfig
# 导入模型相关配置类，其中FNO_Small2d是小尺寸的2D傅里叶神经算子模型配置
from .models import ModelConfig, FNO_Small2d
# 导入优化器和数据分块（Patching）相关配置类
from .opt import OptimizationConfig, PatchingConfig
# 导入Weights & Biases（实验追踪工具）相关配置类
from .wandb import WandbConfig


# Burgers方程数据集配置类
# 继承自ConfigBase，用于统一管理Burgers方程数据集的所有参数
class BurgersDatasetConfig(ConfigBase):
    # 数据集存储的根文件夹路径
    folder: str = "D:\\document\\python\\neural operators\\dataset"
    # 训练时每个批次的样本数量，默认16个样本/批次
    batch_size: int = 16
    # 训练集的样本总数，默认800个训练样本
    n_train: int = 800
    # 测试时使用的批次大小列表（支持多组测试批次配置），默认仅[16]
    test_batch_sizes: List[int] = [16]
    # 测试集的样本数量列表（与test_batch_sizes一一对应），默认仅[400]
    n_tests: List[int] = [400]
    # 空间维度的长度（分辨率）：原始完整分辨率是128x101，这里使用轻量化版本16x17
    spatial_length: int = 16
    # 时间维度的长度（分辨率）：对应轻量化版本的时间步数量
    temporal_length: int = 17
    # 时间维度下采样系数（可选）：若设置为整数n，则每隔n个时间步取一个样本，None表示不下采样
    temporal_subsample: Optional[int] = None
    # 是否对输入数据进行编码（如归一化、特征编码等），默认不编码
    encode_input: bool = False
    # 是否对输出数据进行编码，默认不编码
    encode_output: bool = False
    # 是否包含时间维度的端点（起始/结束时刻）：
    # [True, False] 表示包含起始端点、不包含结束端点，可根据任务需求调整
    include_endpoint: List[bool] = [True, False]


# Burgers方程优化器配置类
# 继承自通用的OptimizationConfig，针对Burgers方程任务定制优化相关参数
class BurgersOptConfig(OptimizationConfig):
    # 训练的总轮数（Epoch），默认训练100轮
    n_epochs: int = 100
    # 训练损失函数类型列表：支持多损失联合训练
    # 可选值包括 "equation"（方程损失，约束模型满足Burgers方程）、
    # "ic"（初始条件损失）、"l2"（L2均方误差损失）等
    training_loss: List[str] = ["equation", "ic", "l2"]
    # 测试时使用的损失函数类型，默认使用L2均方误差评估模型性能
    testing_loss: str = "l2"
    # 优化器的学习率，默认1e-4（0.0001）
    learning_rate: float = 1e-4
    # 权重衰减系数（L2正则化），用于防止过拟合，默认1e-4
    weight_decay: float = 1e-4
    # 模型评估的间隔（按Epoch计），默认每训练1轮就评估一次模型
    eval_interval: int = 1
    # 是否启用混合精度训练（FP16/FP32），默认关闭（使用FP32）
    # 启用后可加速训练、降低显存占用，但需硬件支持（如NVIDIA GPU）
    mixed_precision: bool = False
    # 学习率调度器类型：控制训练过程中学习率的调整策略
    # 可选值："ReduceLROnPlateau"（验证集性能停滞时降低学习率）、
    # "CosineAnnealingLR"（余弦退火学习率）
    scheduler: str = "ReduceLROnPlateau"
    # 仅对ReduceLROnPlateau调度器有效：性能停滞多少轮后降低学习率，默认100轮
    scheduler_patience: int = 100
    # 学习率调整的步长（主要用于CosineAnnealingLR等调度器），默认60轮调整一次
    step_size: int = 60
    # 学习率衰减系数：每次调整时学习率乘以该系数，默认0.5（即降低50%）
    gamma: float = 0.5
    # 多损失函数的聚合策略：将多个训练损失（如equation+ic+l2）合并为总损失的方式
    # 可选值："relobralo"（基于损失相对值的动态加权）、"softadapt"（自适应软加权）
    loss_aggregator: str = "relobralo"


# 整体默认配置类
# 整合所有子配置模块，作为Burgers方程任务的默认配置入口
class Default(ConfigBase):
    # 基线模型的参数数量（可选）：用于对比不同模型的参数量，None表示不设置
    n_params_baseline: Optional[Any] = None
    # 是否启用详细日志输出：True表示打印训练/测试过程中的详细信息
    verbose: bool = True
    # 模型架构类型：默认使用"fno"（傅里叶神经算子）
    arch: str = "fno"
    # 分布式训练配置：初始化分布式训练的默认参数（如多卡、多机配置）
    distributed: DistributedConfig = DistributedConfig()
    # 模型配置：默认使用小尺寸的2D FNO模型配置（FNO_Small2d）
    model: ModelConfig = FNO_Small2d()
    # 优化配置：使用上面定制的Burgers方程优化参数
    opt: OptimizationConfig = BurgersOptConfig()
    # 数据集配置：使用上面定制的Burgers方程数据集参数
    data: BurgersDatasetConfig = BurgersDatasetConfig()
    # 数据分块配置：控制数据分块/拼接的参数（如大尺寸数据的分块训练）
    patching: PatchingConfig = PatchingConfig()
    # Wandb实验追踪配置：默认启用日志记录（log=True），并记录模型输出（log_output=True）
    # Wandb用于可视化训练过程、对比实验结果、保存实验参数等
    wandb: WandbConfig = WandbConfig(log = True, log_output = True)