"""
基于物理信息神经算子（Physics-Informed Neural Operator, PINO）的Burgers方程训练脚本

核心功能：
1. 训练神经算子求解1维时间依赖的Burgers方程（流体力学经典方程）
2. 融合物理信息损失（如方程残差、初始条件约束）和数据损失（如L2/H1损失）
3. 支持两种先进的自适应损失聚合策略：Relobralo和SoftAdapt
4. 集成WandB日志系统，支持训练过程可视化和参数监控
5. 包含完整的训练流程：数据加载、模型构建、优化器配置、训练/评估循环

Burgers方程背景：
1D时间依赖Burgers方程形式：∂u/∂t + u∂u/∂x = ν∂²u/∂x²
其中ν为粘性系数，本脚本中设置为0.01，PINO通过神经网络逼近该方程的解，并利用物理约束提升精度
"""

# 系统模块导入
import sys
# PyTorch核心库：张量计算、自动微分、神经网络
import torch
# WandB：机器学习实验跟踪工具（权重与偏置）
import wandb
# PyTorch函数库：包含各类损失函数、激活函数等
import torch.nn.functional as F

# neuralop库导入（神经算子专用库）
# 损失函数：H1Loss(H1范数损失)、LpLoss(Lp范数损失)、BurgersEqnLoss(Burgers方程物理损失)、ICLoss(初始条件损失)
from neuralop import H1Loss, LpLoss, BurgersEqnLoss, ICLoss, get_model
# 数据集加载：加载1D时间依赖Burgers方程的迷你数据集
from neuralop.data.datasets import load_mini_burgers_1dtime
# 优化器：AdamW（带权重衰减的Adam）
from neuralop.training import AdamW
# 工具函数：WandB密钥获取、模型参数计数、项目根目录获取
from neuralop.utils import get_wandb_api_key, count_model_params, get_project_root
# 损失聚合策略：Relobralo和SoftAdapt（自适应损失加权方法）
from neuralop.losses.meta_losses import Relobralo, SoftAdapt

# ======================== 1. 配置加载 ========================
# 默认配置名称
config_name = "default"
# 从命令行加载配置（zencfg是配置管理库）
from zencfg import make_config_from_cli
import sys

# 添加项目根目录到系统路径（解决模块导入问题）
sys.path.insert(0, "d:\\document\\python\\neural operators")
# 导入Burgers方程PINO的默认配置
from config.burgers_pino_config import Default

# 解析命令行参数并生成配置对象，转换为字典格式
config = make_config_from_cli(Default)
config = config.to_dict()
print(f"加载配置: {config_name}")
print(f"配置内容: {config}")


# ======================== 2. 设备配置 ========================
# 自动检测GPU可用性，优先使用CUDA，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用计算设备: {device}")  # 新增：打印使用的设备，便于调试

# ======================== 3. WandB实验跟踪配置 ========================
if config.wandb.log:
    # 登录WandB（通过密钥认证）
    wandb.login(key=get_wandb_api_key())
    # 设置WandB实验名称（优先使用配置中的名称，否则自动生成）
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        # 自动生成实验名称：配置名+模型架构+层数+模态数+隐藏通道数
        wandb_name = "_".join(
            f"{var}"
            for var in [
                config_name,
                config.model.model_arch,
                config.model.n_layers,
                config.model.n_modes,
                config.model.hidden_channels,
            ]
        )
    # WandB初始化参数
    wandb_init_args = dict(
        config=config,          # 记录实验配置
        name=wandb_name,        # 实验名称
        group=config.wandb.group,  # 实验分组（便于对比不同实验）
        project=config.wandb.project,  # 项目名称
        entity=config.wandb.entity,    # 实体（团队/个人）名称
    )
    # 如果是超参数扫描（sweep），更新配置参数
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]
    # 初始化WandB
    wandb.init(**wandb_init_args)
else:
    wandb_init_args = None

# ======================== 4. 配置信息打印 ========================
# 如果开启详细模式，打印配置信息（便于调试）
if config.verbose:
    print("##### 实验配置详情 ######")
    print(config)
    sys.stdout.flush()  # 强制刷新输出缓冲区，确保即时打印

# ======================== 5. 数据集加载 ========================
# 获取数据集根路径（项目根目录 + 配置中的数据文件夹）
data_path = get_project_root() / config.data.folder
# 加载1D时间依赖的Burgers方程迷你数据集
# 返回值：训练数据加载器、测试数据加载器字典、数据处理器（用于数据预处理/后处理）
train_loader, test_loaders, data_processor = load_mini_burgers_1dtime(
    data_path=data_path,                # 数据存储路径
    n_train=config.data.n_train,        # 训练样本数量
    batch_size=config.data.batch_size,  # 训练批次大小
    n_test=config.data.n_tests[0],      # 测试样本数量
    test_batch_size=config.data.test_batch_sizes[0],  # 测试批次大小
    temporal_subsample=config.data.get("temporal_subsample", 1),  # 时间维度下采样（减少计算量）
    spatial_subsample=config.data.get("spatial_subsample", 1),  # 空间维度下采样
)
print(f"数据集加载完成：训练批次数量={len(train_loader)}, 测试集数量={len(test_loaders)}")

# ======================== 6. 模型构建 ========================
# 根据配置创建神经算子模型（支持FNO、PINO等架构）
model = get_model(config)
# 将模型部署到指定设备（GPU/CPU）
model = model.to(device)
print(f"模型构建完成，部署到{device}")

# ======================== 7. 优化器配置 ========================
# 使用AdamW优化器（带权重衰减，防止过拟合）
optimizer = AdamW(
    model.parameters(),                  # 待优化的模型参数
    lr=config.opt.learning_rate,         # 初始学习率
    weight_decay=config.opt.weight_decay, # 权重衰减系数（L2正则化）
)

# ======================== 8. 学习率调度器配置 ========================
# 根据配置选择学习率调度策略（动态调整学习率，提升训练效果）
if config.opt.scheduler == "ReduceLROnPlateau":
    #  Plateau调度器：验证损失停止下降时降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.opt.gamma,               # 学习率衰减因子（如0.5表示减半）
        patience=config.opt.scheduler_patience, # 等待轮数（损失不下降则衰减）
        mode="min",                             # 最小化损失
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
    # 未知调度器类型，抛出异常
    raise ValueError(f"不支持的学习率调度器: {config.opt.scheduler}")

# ======================== 9. 损失函数配置 ========================
# 定义基础损失函数（物理信息+数据驱动损失结合）
l2_loss = LpLoss(d=2, p=2)  # L2范数损失（数据拟合损失，d=2表示时空二维）
h1_loss = H1Loss(d=2)       # H1范数损失（包含函数值和一阶导数的损失，更严格的拟合）
ic_loss = ICLoss()          # 初始条件损失（约束模型满足Burgers方程的初始条件）
# Burgers方程物理损失（核心：强制模型输出满足Burgers方程的物理规律）
equation_loss = BurgersEqnLoss(
    method=config.opt.get("pino_method", "fdm"),  # 数值方法（fdm=有限差分法）
    visc=0.01,                                    # Burgers方程粘性系数
    loss=F.mse_loss                               # 基础损失函数（均方误差）
)

# 损失函数映射字典（便于根据配置快速选择）
loss_map = {
    "l2": l2_loss,          # 数据拟合损失
    "h1": h1_loss,          # 高阶数据拟合损失
    "ic": ic_loss,          # 初始条件物理约束
    "equation": equation_loss  # Burgers方程残差约束（核心物理损失）
}

# 根据配置选择训练用的损失函数列表
training_losses = [loss_map[name] for name in config.opt.training_loss]
print(f"训练使用的损失函数: {config.opt.training_loss}")

# ======================== 10. 自适应损失聚合策略 ========================
# 传统损失加权是固定系数，自适应聚合可根据训练动态调整各损失权重
if config.opt.loss_aggregator.lower() == "relobralo":
    # Relobralo策略：基于损失梯度的自适应加权
    train_loss = Relobralo(
        num_losses=len(training_losses),  # 损失函数数量
        params=model.parameters(),        # 模型参数（用于计算梯度）
        alpha=0.5,                        # 权重更新系数
        beta=0.9,                         # 动量系数
        tau=1.0,                          # 温度系数
    )
elif config.opt.loss_aggregator.lower() == "softadapt":
    # SoftAdapt策略：基于损失值的自适应加权
    train_loss = SoftAdapt(
        num_losses=len(training_losses),  # 损失函数数量
        params=model.parameters(),        # 模型参数
    )
else:
    # 未知聚合策略，抛出异常
    raise ValueError(
        f"不支持的损失聚合策略: {config.opt.loss_aggregator}。请使用'relobralo'或'softadapt'."
    )

# 评估用损失函数（仅用于测试集评估，不参与训练）
eval_losses = {"h1": h1_loss, "l2": l2_loss}

# ======================== 11. 模型信息打印 ========================
if config.verbose:
    print("\n### 模型结构 ###\n", model)
    print("\n### 优化器 ###\n", optimizer)
    print("\n### 学习率调度器 ###\n", scheduler)
    print("\n### 损失函数 ###")
    print(f"\n * 训练损失（聚合）: {train_loss}")
    print(f"\n * 测试损失: {eval_losses}")
    print(f"\n### 开始训练...\n")
    sys.stdout.flush()

# 统计模型参数数量并记录
if config.verbose:
    n_params = count_model_params(model)
    print(f"\n模型总参数数量: {n_params:,}")  # 格式化输出（千分位）
    sys.stdout.flush()
if config.wandb.log:
    # 将参数数量记录到WandB
    wandb.log({"n_params": n_params}, commit=False)
    # 监控模型梯度、参数等（便于调试）
    wandb.watch(model)

# ======================== 12. 主训练循环 ========================
# 设置模型为训练模式（启用Dropout、BatchNorm等训练层）
model.train()
for epoch in range(config.opt.n_epochs):
    # 每个epoch开始时重置训练模式（评估后恢复）
    model.train()
    # 记录当前epoch的总损失
    train_losses = []
    # 记录每个损失分量的数值（便于监控）
    loss_values = {name: [] for name in config.opt.training_loss}

    # 遍历训练批次
    for batch_idx, sample in enumerate(train_loader):
        # ======================== 批次数据预处理 ========================
        # 将所有张量数据移到指定设备（GPU/CPU）并转换为float32（避免精度问题）
        sample = {
            k: v.to(device).float() if torch.is_tensor(v) else v
            for k, v in sample.items()
        }

        # 清空优化器梯度（set_to_none=True比zero_grad()更高效）
        optimizer.zero_grad(set_to_none=True)

        # 数据处理器预处理（如归一化、维度调整等）
        if data_processor is not None:
            sample = data_processor.preprocess(sample)
            # 预处理后再次确认设备和数据类型
            sample = {
                k: v.to(device).float() if torch.is_tensor(v) else v
                for k, v in sample.items()
            }

        # ======================== 前向传播 ========================
        # 模型前向预测（**sample解包输入数据，如初始条件、坐标等）
        pred = model(**sample)

        # 数据处理器后处理（如反归一化，恢复原始尺度）
        if data_processor is not None:
            pred, sample = data_processor.postprocess(pred, sample)
            # 后处理后再次确认设备和数据类型
            sample = {
                k: v.to(device).float() if torch.is_tensor(v) else v
                for k, v in sample.items()
            }

        # ======================== 损失计算 ========================
        # 计算每个损失分量的数值
        loss_vals = {}
        for loss_name in config.opt.training_loss:
            if loss_name == "equation":
                # 物理损失需要额外传入坐标信息（计算方程残差）
                loss_val = loss_map[loss_name](pred, x=sample["x"])
            else:
                # 数据损失：预测值 vs 真实值（sample["y"]是标签）
                loss_val = loss_map[loss_name](pred, sample["y"])

            # 记录当前批次的损失值
            loss_vals[loss_name] = loss_val
            loss_values[loss_name].append(loss_val.item())

        # 自适应聚合损失（核心：动态加权各损失分量）
        total_loss, weights = train_loss(loss_vals, step=epoch)

        # ======================== 反向传播与优化 ========================
        # 损失反向传播（计算梯度）
        total_loss.backward()
        # 优化器更新参数
        optimizer.step()

        # 记录当前批次的总损失
        train_losses.append(total_loss.item())

    # ======================== Epoch结束后处理 ========================
    # 计算当前epoch的平均训练损失
    avg_train_loss = sum(train_losses) / len(train_losses)
    # 计算每个损失分量的平均数值
    avg_losses = {name: sum(vals) / len(vals) for name, vals in loss_values.items()}

    # 打印训练进度
    if config.verbose:
        print(f"\n===== Epoch {epoch+1}/{config.opt.n_epochs} =====")
        print(f"平均训练总损失: {avg_train_loss:.6f}")
        print("各损失分量平均值:")
        for name, avg_val in avg_losses.items():
            print(f"  {name.upper()} 损失: {avg_val:.6f}")

        # 打印自适应损失权重（便于监控权重变化）
        if hasattr(train_loss, "weights"):
            print("自适应损失权重:")
            for i, name in enumerate(config.opt.training_loss):
                print(f"  {name.upper()} 权重: {weights[i]:.6f}")
        sys.stdout.flush()

    # ======================== 周期性测试集评估 ========================
    eval_losses_dict = {}
    # 每隔指定轮数（eval_interval）评估一次测试集
    if epoch % config.opt.eval_interval == 0:
        # 设置模型为评估模式（禁用Dropout、BatchNorm等）
        model.eval()

        # 禁用梯度计算（加速推理，节省显存）
        with torch.no_grad():
            # 遍历所有测试集（支持多个测试集）
            for test_name, test_loader in test_loaders.items():
                test_losses = []
                for sample in test_loader:
                    # 测试数据预处理（同训练集）
                    sample = {
                        k: v.to(device).float() if torch.is_tensor(v) else v
                        for k, v in sample.items()
                    }

                    if data_processor is not None:
                        sample = data_processor.preprocess(sample)
                        sample = {
                            k: v.to(device).float() if torch.is_tensor(v) else v
                            for k, v in sample.items()
                        }

                    # 模型预测
                    pred = model(**sample)

                    # 后处理
                    if data_processor is not None:
                        pred, sample = data_processor.postprocess(pred, sample)
                        sample = {
                            k: v.to(device).float() if torch.is_tensor(v) else v
                            for k, v in sample.items()
                        }

                    # 计算测试损失（L2）
                    l2_test_loss = l2_loss(pred.float(), sample["y"].float())
                    test_losses.append(l2_test_loss.item())

                # 计算测试集平均损失
                avg_test_loss = sum(test_losses) / len(test_losses)
                eval_losses_dict[f"{test_name}_loss"] = avg_test_loss

                # 打印测试损失
                if config.verbose:
                    print(f"\n{test_name} 测试集 L2 损失: {avg_test_loss:.6f}")

    # ======================== 学习率更新 ========================
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        # Plateau调度器：基于测试损失（无测试损失则用训练损失）更新
        scheduler.step(eval_losses_dict.get("test_loss", avg_train_loss))
    else:
        # 其他调度器：按epoch更新
        scheduler.step()

    # ======================== WandB日志记录 ========================
    if config.wandb.log:
        # 构建日志字典
        log_dict = {
            "train_loss": avg_train_loss,                # 平均训练总损失
            "learning_rate": optimizer.param_groups[0]["lr"],  # 当前学习率
        }

        # 记录各损失分量
        for name, avg_val in avg_losses.items():
            log_dict[f"train_{name}_loss"] = avg_val

        # 记录自适应损失权重
        if hasattr(train_loss, "weights"):
            for i, name in enumerate(config.opt.training_loss):
                log_dict[f"weight_{name}"] = weights[i]

        # 记录测试损失
        if eval_losses_dict:
            log_dict.update(eval_losses_dict)

        # 写入WandB
        wandb.log(log_dict)

    # ======================== GPU内存清理 ========================
    # 每10个epoch清理一次GPU缓存（防止显存泄漏）
    if device.type == "cuda" and epoch % 10 == 0:
        torch.cuda.empty_cache()

# ======================== 训练结束 ========================
# 结束WandB日志记录
if config.wandb.log:
    wandb.finish()

print("\n训练完成！")

# ======================== 训练结束 - 保存模型 ========================
# 定义模型保存路径（建议包含实验名称，便于区分）
import os
save_dir = os.path.join(get_project_root(), "trained_models", wandb_name if config.wandb.log else "burgers_pino_model")
os.makedirs(save_dir, exist_ok=True)

# 保存核心内容：模型权重、配置、数据处理器（关键！保证预处理/后处理一致）
model_save_path = os.path.join(save_dir, "model_weights.pth")
config_save_path = os.path.join(save_dir, "config.pth")
processor_save_path = os.path.join(save_dir, "data_processor.pth")

# 1. 保存模型权重（仅保存参数，节省空间）
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),  # 可选：保存优化器状态，便于续训
    'epoch': config.opt.n_epochs,                    # 可选：保存训练轮数
}, model_save_path)

# 2. 保存配置（推理时需要一致的模型结构/参数）
torch.save(config, config_save_path)

# 3. 保存数据处理器（关键！推理时预处理/后处理必须和训练一致）
if data_processor is not None:
    torch.save(data_processor, processor_save_path)

print(f"模型已保存至：{save_dir}")