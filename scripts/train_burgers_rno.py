"""
使用递归神经算子（Recurrent Neural Operator, RNO）训练Burgers方程的训练脚本。

本脚本针对一维时间依赖的Burgers方程，基于仓库内的小型数据集训练RNO模型。
Burgers方程是流体力学中经典的非线性偏微分方程，常用于验证神经算子的性能。
核心功能：
1. 适配数据集格式与RNO模型输入格式的差异
2. 提取目标序列的最后时间步进行单步监督训练
3. 完整的训练流程（数据加载、模型构建、优化器配置、训练监控、WandB日志）
"""

# 系统模块导入：用于路径处理和系统配置
import sys
import os

# 核心深度学习库
import torch
import torch.nn as nn

# 实验跟踪工具：WandB (Weights & Biases) 用于监控训练过程
import wandb

# 配置管理工具：从命令行解析配置
from zencfg import make_config_from_cli

# 将项目根目录添加到系统路径，确保能导入项目内部模块
# 获取当前脚本的绝对路径的目录名
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
# 项目根目录为当前脚本目录的父目录
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# 若根目录未在sys.path中，则添加（避免导入模块时出现找不到的错误）
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 项目内部模块导入
from config.burgers_rno_config import Default  # 默认配置文件
from neuralop.utils import get_wandb_api_key, count_model_params  # 工具函数：WandB密钥、参数计数
from neuralop.training import AdamW, Trainer  # 训练相关：优化器、训练器
from neuralop.losses import LpLoss  # 损失函数：Lp范数损失（这里用L2）
from neuralop.models import get_model  # 模型构建函数：根据配置获取RNO模型
from neuralop.data.datasets import load_mini_burgers_1dtime  # 数据集加载：小型1D时间依赖Burgers数据集


class RNOTimeAdapter(nn.Module):
    """
    数据格式适配器：统一数据集输出格式与RNO模型输入格式的差异。
    
    核心问题：
    - mini_burgers_1dtime数据集返回的批次形状为 (B, C, T, X)，其中时间维度在索引2
      （B=批次大小, C=通道数, T=时间步, X=空间维度）
    - RNO模型期望的输入形状为 (B, T, C, X)，时间维度在索引1
    本适配器通过维度置换解决格式不匹配问题，并返回最后时间步的预测结果。
    
    设计目的：
    Trainer要求模型满足标准的forward(x, y)签名，且输出需匹配目标格式；
    而原始RNO模型返回预测值+隐藏状态，且输入格式不同，因此需要该包装器适配。

    参数
    ----------
    core : nn.Module
        待包装的核心RNO模型（实际执行序列建模的模型）
    """
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core  # 保存核心RNO模型

    def forward(self, x, y=None):
        """
        前向传播函数：完成维度变换并调用核心模型。
        
        输入
        -----
        x : torch.Tensor
            来自数据集的输入，形状为 (B, C, T, X)
        y : torch.Tensor, optional
            目标值（Trainer要求的参数，实际未使用）
        
        返回
        -------
        torch.Tensor
            最后时间步的预测结果，形状为 (B, C, X)
        """
        # 维度置换：(B, C, T, X) -> (B, T, C, X)，适配RNO输入格式
        # contiguous()确保张量内存连续，避免后续操作报错
        x_rno = x.permute(0, 2, 1, 3).contiguous()
        # 调用核心RNO模型得到预测
        pred = self.core(x_rno)
        return pred


class LastFrameDataProcessor(nn.Module):
    """
    数据处理器包装器：从目标序列中提取最后时间帧，用于单步监督训练。
    
    核心问题：
    - Burgers数据集提供的目标是完整的时间序列，形状为 (B, C, T, X)
    - 我们训练RNO仅预测最后一个时间步，因此需要将目标转换为 (B, C, X) 形状
      以适配L2损失的计算（损失需要预测值和目标值维度匹配）
    
    设计目的：
    包装现有数据处理器，在预处理阶段提取目标的最后时间步，不修改原有处理器的核心逻辑。

    参数
    ----------
    base : nn.Module or object
        待包装的基础数据处理器（如DefaultDataProcessor）
        其preprocess/postprocess方法会在提取最后帧之前/之后调用
    """
    def __init__(self, base):
        super().__init__()
        self.base = base  # 保存基础数据处理器
        self.device = "cpu"  # 默认设备为CPU

    def to(self, device):
        """
        重写to方法：将处理器移至指定设备（兼容基础处理器的设备迁移）。
        
        参数
        ----------
        device : torch.device or str
            目标设备（如"cuda"、"cpu"）
        
        返回
        -------
        self : LastFrameDataProcessor
            迁移后的处理器实例
        """
        self.device = device
        # 如果基础处理器有to方法，则同步迁移
        if hasattr(self.base, "to"):
            self.base = self.base.to(device)
        return self

    def preprocess(self, data_dict, batched=True):
        """
        预处理函数：先调用基础处理器的预处理，再提取目标的最后时间步。
        
        参数
        ----------
        data_dict : dict
            包含输入(x)和目标(y)的字典，格式为{"x": tensor, "y": tensor}
        batched : bool, default=True
            标识数据是否为批次形式（兼容基础处理器的参数）
        
        返回
        -------
        data_dict : dict
            预处理后的字典，目标y的形状变为 (B, C, X)
        """
        # 先调用基础处理器的预处理逻辑
        if hasattr(self.base, "preprocess"):
            data_dict = self.base.preprocess(data_dict, batched=batched)
        # 仅对最后时间步进行监督：提取y的最后一个时间维度
        if torch.is_tensor(data_dict["y"]):
            # 索引说明：[:, :, -1, :] 表示保留批次、通道、空间维度，取最后一个时间步
            data_dict["y"] = data_dict["y"][:, :, -1, :]
        return data_dict

    def postprocess(self, output, data_dict):
        """
        后处理函数：直接调用基础处理器的后处理逻辑（无额外操作）。
        
        参数
        ----------
        output : torch.Tensor
            模型的输出结果
        data_dict : dict
            预处理后的数据集字典
        
        返回
        -------
        tuple
            后处理后的输出和数据字典
        """
        if hasattr(self.base, "postprocess"):
            return self.base.postprocess(output, data_dict)
        return output, data_dict


def main():
    """
    主训练函数：整合所有模块，完成从配置加载到模型训练的完整流程。
    核心步骤：
    1. 加载配置（命令行+默认配置）
    2. 设备初始化（CUDA/CPU）
    3. WandB日志配置
    4. 加载数据集和数据处理器
    5. 构建模型（RNO+格式适配器）
    6. 配置优化器、学习率调度器、损失函数
    7. 初始化训练器并启动训练
    """
    # 加载配置：从命令行解析参数，基于Default配置生成最终配置
    config = make_config_from_cli(Default)
    # 转换为字典格式，方便后续访问
    config = config.to_dict()

    # 设备配置：优先使用CUDA（GPU），无GPU则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # WandB日志配置：用于可视化训练过程（损失、学习率、模型参数等）
    if config.wandb.log:
        # 登录WandB（通过项目内的密钥获取函数）
        wandb.login(key=get_wandb_api_key())
        # 设置WandB运行名称：优先使用配置中的名称，否则自动生成
        if config.wandb.name:
            wandb_name = config.wandb.name
        else:
            # 自动生成名称：架构_层数_模态数_隐藏通道数
            wandb_name = "_".join(
                f"{var}" for var in [
                    config["arch"],          # 模型架构（RNO）
                    config.model.n_layers,   # 网络层数
                    config.model.n_modes,    # 傅里叶模态数
                    config.model.hidden_channels,  # 隐藏层通道数
                ]
            )
        # WandB初始化参数
        wandb_init_args = dict(
            config=config,          # 记录训练配置
            name=wandb_name,        # 运行名称
            group=config.wandb.group,  # 实验分组
            project=config.wandb.project,  # 项目名称
            entity=config.wandb.entity,    # 团队/个人实体
        )
        # 支持WandB超参数扫描（sweep）
        if config.wandb.sweep:
            for key in wandb.config.keys():
                config.params[key] = wandb.config[key]
        # 初始化WandB
        wandb.init(**wandb_init_args)
    else:
        # 不启用WandB时，初始化参数设为None
        wandb_init_args = None

    # 打印配置信息（调试用）
    if config.verbose:
        print("##### 训练配置 ######")
        print(config)
        # 强制刷新输出缓冲区，确保配置信息即时打印
        sys.stdout.flush()

    # 数据加载：加载小型1D时间依赖Burgers数据集
    data_path = os.path.join(PROJECT_ROOT, config.data.folder)  # 数据集根路径
    train_loader, test_loaders, data_processor = load_mini_burgers_1dtime(
        data_path=data_path,                # 数据集路径
        n_train=config.data.n_train,        # 训练集样本数
        batch_size=config.data.batch_size,  # 训练批次大小
        n_test=config.data.n_tests[0],      # 测试集样本数
        test_batch_size=config.data.test_batch_sizes[0],  # 测试批次大小
        temporal_subsample=config.data.get("temporal_subsample", 1),  # 时间维度下采样（减少时间步）
        spatial_subsample=config.data.get("spatial_subsample", 1),    # 空间维度下采样（减少空间点）
    )

    # 模型实例化与包装
    core_model = get_model(config)  # 根据配置构建核心RNO模型
    # 包装核心模型：适配数据格式，移至指定设备（GPU/CPU）
    model = RNOTimeAdapter(core_model).to(device)

    # 包装数据处理器：将目标转换为最后时间步
    if data_processor is not None:
        data_processor = LastFrameDataProcessor(data_processor).to(device)

    # 优化器配置：使用AdamW（带权重衰减的Adam）
    optimizer = AdamW(
        model.parameters(),                # 待优化的模型参数
        lr=config.opt.learning_rate,       # 初始学习率
        weight_decay=config.opt.weight_decay,  # 权重衰减（防止过拟合）
    )

    # 学习率调度器配置：根据训练情况调整学习率
    if config.opt.scheduler == "ReduceLROnPlateau":
        # 平台调度器：验证损失停止下降时降低学习率
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.opt.gamma,               # 学习率衰减因子（如0.5）
            patience=config.opt.scheduler_patience,  # 多少轮无提升后衰减
            mode="min",                            # 基于最小化损失调整
        )
    elif config.opt.scheduler == "CosineAnnealingLR":
        # 余弦退火调度器：学习率按余弦函数周期性变化
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.opt.scheduler_T_max  # 余弦周期
        )
    elif config.opt.scheduler == "StepLR":
        # 步长调度器：每隔指定轮数衰减学习率
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
        )
    else:
        # 不支持的调度器类型，抛出异常
        raise ValueError(f"不支持的学习率调度器类型: {config.opt.scheduler}")

    # 损失函数配置：L2损失（d=2表示空间维度为2？此处实际为1D，d=1也可，LpLoss兼容）
    l2_loss = LpLoss(d=2, p=2)
    # 评估损失字典：训练和测试均使用L2损失
    eval_losses = {"l2": l2_loss}
    train_loss = l2_loss

    # 打印训练组件信息（调试用）
    if config.verbose:
        print("\n### 模型结构 ###\n", model)
        print("\n### 优化器 ###\n", optimizer)
        print("\n### 学习率调度器 ###\n", scheduler)
        print("\n### 损失函数 ###")
        print(f"\n * 训练损失: {train_loss}")
        print(f"\n * 测试损失: {eval_losses}")
        print(f"\n### 开始训练...\n")
        sys.stdout.flush()

    # 训练器初始化：封装训练逻辑（前向、反向、评估、日志）
    trainer = Trainer(
        model=model,                          # 待训练的模型
        n_epochs=config.opt.n_epochs,         # 训练轮数
        device=device,                        # 训练设备
        data_processor=data_processor,        # 数据处理器（含最后时间步提取）
        mixed_precision=config.opt.mixed_precision,  # 混合精度训练（加速、省显存）
        wandb_log=config.wandb.log,           # 是否启用WandB日志
        eval_interval=config.opt.eval_interval,  # 评估间隔（每多少轮测试一次）
        log_output=config.wandb.log_output,   # 是否记录模型输出
        use_distributed=config.distributed.use_distributed,  # 是否分布式训练
        verbose=config.verbose,               # 是否打印详细日志
    )

    # 统计模型参数数量（监控模型复杂度）
    n_params = count_model_params(model)
    if config.verbose:
        print(f"\n模型总参数量: {n_params}")
        sys.stdout.flush()
    if config.wandb.log:
        # 将参数数量记录到WandB
        wandb.log({"n_params": n_params}, commit=False)
        # WandB监控模型梯度、参数分布等
        wandb.watch(model)

    # 启动训练
    trainer.train(
        train_loader=train_loader,    # 训练数据加载器
        test_loaders=test_loaders,    # 测试数据加载器（列表形式）
        optimizer=optimizer,          # 优化器
        scheduler=scheduler,          # 学习率调度器
        regularizer=False,            # 是否使用正则化（此处禁用）
        training_loss=train_loss,     # 训练损失函数
        eval_losses=eval_losses,      # 评估损失函数字典
    )

    # 训练结束，关闭WandB
    if config.wandb.log:
        wandb.finish()


# 脚本入口：仅当直接运行该脚本时执行main函数
if __name__ == "__main__":
    main()