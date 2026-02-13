"""
基于神经算子(Neural Operator)的2D达西流方程训练脚本

【脚本核心功能】
该脚本针对2D达西流方程（描述多孔介质中流体流动的偏微分方程）训练神经算子模型，
核心目标是用数据驱动的神经算子替代传统数值方法（如有限元、有限差分）求解达西流方程。
脚本支持两大关键特性：
1. 分布式训练：利用多GPU/多节点提升训练效率
2. 多网格补丁(MGPatching)：对高分辨率数据进行分块处理，解决显存限制并提升高分辨率数据的训练效果

【达西流背景】
达西流方程是描述流体在多孔介质中流动的经典偏微分方程，广泛应用于石油工程、地下水模拟等领域，
神经算子作为一种高效的算子学习方法，能够直接学习输入（渗透率场）到输出（压力场/流速场）的映射关系。
"""

# 导入路径处理库，用于统一不同操作系统的文件路径
from pathlib import Path
# 系统相关操作库，用于路径插入、标准输出刷新等
import sys

# PyTorch核心库，用于张量计算和深度学习
import torch
# PyTorch数据加载核心模块
from torch.utils.data import DataLoader, DistributedSampler
# Weights & Biases，用于实验跟踪、可视化和日志记录
import wandb

# neuralop库核心模块：损失函数、训练器、模型构建、数据集加载等
from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.data.datasets import load_darcy_flow_small  # 加载达西流小型数据集
from neuralop.data.transforms.data_processors import MGPatchingDataProcessor  # 多网格补丁数据处理器
from neuralop.training import setup, AdamW  # 分布式训练初始化、带权重衰减的Adam优化器
from neuralop.mpu.comm import get_local_rank  # 获取分布式训练中的本地进程rank
from neuralop.utils import get_wandb_api_key, count_model_params  # WandB密钥获取、模型参数量统计

# 配置文件相关导入
from zencfg import make_config_from_cli  # 从命令行解析配置
import sys

# 将当前目录加入系统路径，确保能导入自定义配置文件
sys.path.insert(0, "./")
from config.darcy_config import Default  # 达西流训练的默认配置类

# ============================ 1. 配置解析 ============================
# 从命令行参数解析配置（优先使用命令行参数，未指定则使用Default类的默认值）
# 最终转换为字典格式，方便后续调用
config = make_config_from_cli(Default)
config = config.to_dict()

# ============================ 2. 分布式训练环境初始化 ============================
# setup函数完成以下核心工作：
# - 初始化分布式训练环境（如果启用）
# - 确定训练设备（GPU/CPU）
# - 判断当前进程是否为日志主进程（仅主进程打印日志、记录WandB）
device, is_logger = setup(config)

# ============================ 3. WandB实验日志配置 ============================
wandb_args = None
# 仅当启用WandB且当前进程是日志主进程时初始化WandB
if config.wandb.log and is_logger:
    # 使用本地密钥登录WandB（需提前配置WANDB_API_KEY环境变量）
    wandb.login(key=get_wandb_api_key())
    # 自定义WandB实验名称，未指定则用模型关键参数拼接
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                "darcy",
                config.model.model_arch,      # 模型架构（如FNO、TFNO）
                config.model.n_layers,        # 模型层数
                config.model.n_modes,         # 傅里叶模态数（FNO核心参数）
                config.model.hidden_channels, # 隐藏层通道数
            ]
        )
    # 构造WandB初始化参数
    wandb_args = dict(
        config=config,          # 记录所有实验配置
        name=wandb_name,        # 实验名称
        group=config.wandb.group,  # 实验分组（用于对比不同实验）
        project=config.wandb.project,  # 项目名称
        entity=config.wandb.entity,    # WandB实体（个人/团队）
    )
    # 如果是WandB超参数扫面（sweep），更新配置中的参数
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]
    # 初始化WandB
    wandb.init(**wandb_args)

# 控制日志打印：仅主进程且启用verbose时打印详细信息
config.verbose = config.verbose and is_logger

# 打印配置信息（仅主进程）
if config.verbose and is_logger:
    print(f"##### 实验配置 #####\n")
    print(config)
    # 强制刷新标准输出，确保配置信息及时打印（分布式训练中重要）
    sys.stdout.flush()

# ============================ 4. 加载达西流数据集 ============================
# 解析数据根目录（支持~符号展开为用户主目录）
data_root = Path(config.data.folder).expanduser()
# 加载小型达西流数据集（预处理后的标准化数据）
# 返回值说明：
# - train_loader: 训练数据加载器
# - test_loaders: 字典形式的测试数据加载器（不同分辨率）
# - data_processor: 数据处理器（包含输入/输出归一化、维度调整等）
train_loader, test_loaders, data_processor = load_darcy_flow_small(
    data_root=data_root,               # 数据存储根目录
    n_train=config.data.n_train,       # 训练样本数量
    batch_size=config.data.batch_size, # 训练批次大小
    test_resolutions=config.data.test_resolutions,  # 测试数据分辨率（如64x64, 128x128）
    n_tests=config.data.n_tests,       # 每个分辨率的测试样本数
    test_batch_sizes=config.data.test_batch_sizes,  # 测试批次大小
    encode_input=False,                # 不编码输入（直接使用原始张量）
    encode_output=False,               # 不编码输出（直接使用原始张量）
)

# ============================ 5. 模型初始化 ============================
# 根据配置自动构建神经算子模型（支持FNO、TFNO、MLP等架构）
model = get_model(config)

# ============================ 6. 多网格补丁数据处理器（高分辨率数据专用） ============================
# 当补丁层级>0时，将默认数据处理器替换为多网格补丁处理器
# 核心作用：将高分辨率数据分块（patch）处理，降低单批次显存占用，提升训练效率
if config.patching.levels > 0:
    data_processor = MGPatchingDataProcessor(
        model=model,                          # 关联训练的模型
        in_normalizer=data_processor.in_normalizer,  # 输入归一化器（复用原有归一化参数）
        out_normalizer=data_processor.out_normalizer, # 输出归一化器
        padding_fraction=config.patching.padding,     # 补丁重叠比例（防止拼接边缘误差）
        stitching=config.patching.stitching,         # 拼接策略（如平均拼接、重叠拼接）
        levels=config.patching.levels,               # 多网格层级数
        use_distributed=config.distributed.use_distributed,  # 是否启用分布式
        device=device,                             # 计算设备
    )

# ============================ 7. 分布式训练数据加载器重构 ============================
# 分布式训练中，需要用DistributedSampler确保每个进程处理不重叠的数据集子集
if config.distributed.use_distributed:
    # 重构训练数据加载器
    train_db = train_loader.dataset  # 提取训练数据集（剥离DataLoader）
    # 创建分布式采样器（按rank分配数据）
    train_sampler = DistributedSampler(train_db, rank=get_local_rank())
    train_loader = DataLoader(
        dataset=train_db, 
        batch_size=config.data.batch_size, 
        sampler=train_sampler  # 替换默认采样器为分布式采样器
    )
    # 重构测试数据加载器（每个分辨率单独处理）
    for (res, loader), batch_size in zip(
        test_loaders.items(), config.data.test_batch_sizes
    ):
        test_db = loader.dataset
        test_sampler = DistributedSampler(test_db, rank=get_local_rank())
        test_loaders[res] = DataLoader(
            dataset=test_db, 
            batch_size=batch_size, 
            shuffle=False,  # 测试集不打乱
            sampler=test_sampler
        )

# ============================ 8. 优化器与学习率调度器 ============================
# 初始化AdamW优化器（带权重衰减的Adam，缓解过拟合）
optimizer = AdamW(
    model.parameters(),
    lr=config.opt.learning_rate,    # 初始学习率
    weight_decay=config.opt.weight_decay,  # 权重衰减系数（L2正则）
)

# 根据配置选择学习率调度器（动态调整学习率）
if config.opt.scheduler == "ReduceLROnPlateau":
    #  Plateau调度器：验证损失停止下降时降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.opt.gamma,               # 学习率衰减因子（如0.5）
        patience=config.opt.scheduler_patience, # 等待轮数（损失不下降则衰减）
        mode="min",                            # 最小化损失
    )
elif config.opt.scheduler == "CosineAnnealingLR":
    # 余弦退火调度器：学习率按余弦函数周期性变化
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.opt.scheduler_T_max  # 余弦周期（训练轮数）
    )
elif config.opt.scheduler == "StepLR":
    # 步长调度器：每step_size轮衰减一次学习率
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
    )
else:
    # 不支持的调度器类型抛出异常
    raise ValueError(f"不支持的学习率调度器: {config.opt.scheduler}")

# ============================ 9. 损失函数定义 ============================
# L2损失（LpLoss，p=2）：衡量预测值与真实值的均方误差
l2loss = LpLoss(d=2, p=2)  # d=2表示2D数据，p=2表示L2范数
# H1损失：L2损失 + 梯度的L2损失（更关注解的光滑性，适合偏微分方程求解）
h1loss = H1Loss(d=2)        # d=2表示2D数据

# 选择训练损失函数
if config.opt.training_loss == "l2":
    train_loss = l2loss
elif config.opt.training_loss == "h1":
    train_loss = h1loss
else:
    raise ValueError(
        f"训练损失函数配置错误: {config.opt.training_loss} "
        f'仅支持["l2", "h1"]'
    )

# 验证损失函数（同时监控H1和L2损失）
eval_losses = {"h1": h1loss, "l2": l2loss}

# 打印模型、优化器、调度器、损失函数信息（仅主进程）
if config.verbose and is_logger:
    print("\n### 模型结构 ###\n", model)
    print("\n### 优化器配置 ###\n", optimizer)
    print("\n### 学习率调度器 ###\n", scheduler)
    print("\n### 损失函数 ###")
    print(f"\n * 训练损失: {train_loss}")
    print(f"\n * 验证损失: {eval_losses}")
    print(f"\n### 开始训练...\n")
    sys.stdout.flush()

# ============================ 10. 训练器初始化 ============================
# neuralop封装的Trainer类，整合训练循环、验证、日志、分布式训练等逻辑
trainer = Trainer(
    model=model,                          # 待训练模型
    n_epochs=config.opt.n_epochs,         # 训练总轮数
    device=device,                        # 计算设备
    data_processor=data_processor,        # 数据处理器（归一化、补丁处理等）
    mixed_precision=config.opt.mixed_precision,  # 混合精度训练（加速训练、降低显存）
    wandb_log=config.wandb.log,           # 是否启用WandB日志
    eval_interval=config.opt.eval_interval,     # 验证间隔（每N轮验证一次）
    log_output=config.wandb.log_output,   # 是否记录模型输出（可视化）
    use_distributed=config.distributed.use_distributed,  # 分布式训练开关
    verbose=config.verbose and is_logger, # 日志详细程度
)

# ============================ 11. 模型参数量统计与日志 ============================
if is_logger:
    # 统计模型可训练参数总数
    n_params = count_model_params(model)

    # 打印参数量（仅主进程）
    if config.verbose:
        print(f"\n模型参数量: {n_params}")
        sys.stdout.flush()

    # 将参数量记录到WandB
    if config.wandb.log:
        to_log = {"n_params": n_params}
        # 如果设置了参数量基线，计算压缩比和空间节省率
        if config.n_params_baseline is not None:
            to_log["n_params_baseline"] = config.n_params_baseline
            to_log["compression_ratio"] = config.n_params_baseline / n_params
            to_log["space_savings"] = 1 - (n_params / config.n_params_baseline)
        # 记录到WandB（commit=False表示不立即提交，后续可追加）
        wandb.log(to_log, commit=False)
        # 监控模型梯度、参数等（WandB可视化）
        wandb.watch(model)

# ============================ 12. 启动训练 ============================
trainer.train(
    train_loader=train_loader,    # 训练数据加载器
    test_loaders=test_loaders,    # 测试数据加载器（多分辨率）
    optimizer=optimizer,          # 优化器
    scheduler=scheduler,          # 学习率调度器
    regularizer=False,            # 是否启用正则化（当前禁用）
    training_loss=train_loss,     # 训练损失函数
    eval_losses=eval_losses,      # 验证损失函数（字典形式）
)

# ============================ 13. 结束WandB日志 ============================
if config.wandb.log and is_logger:
    wandb.finish()