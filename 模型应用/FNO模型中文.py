# 导入偏方法创建工具，用于动态修改类的__init__方法默认参数
from functools import partialmethod
# 导入类型注解相关模块，用于定义清晰的类型提示
from typing import Tuple, List, Union, Literal

# 定义数值类型别名，可表示浮点数或整数
Number = Union[float, int]

# 导入PyTorch核心库：张量操作、神经网络层、函数式接口（激活/损失等）
import torch
import torch.nn as nn
import torch.nn.functional as F

# 设置警告过滤器：仅显示每个UserWarning一次，避免重复警告干扰
import warnings
warnings.filterwarnings("once", category=UserWarning)

# 从自定义模块导入FNO核心组件
from ..layers.embeddings import GridEmbeddingND, GridEmbedding2D  # N维/2维网格位置编码层
from ..layers.spectral_convolution import SpectralConv            # 谱卷积层（FNO核心）
from ..layers.padding import DomainPadding                        # 域填充层（缓解边界效应）
from ..layers.fno_block import FNOBlocks                          # FNO块组合层（多层谱卷积+MLP）
from ..layers.channel_mlp import ChannelMLP                      # 通道混合MLP层
from ..layers.complex import ComplexValued                        # 复数值适配层（处理复数数据）
from .base_model import BaseModel                                # 模型基类（提供通用基础功能）


class FNO(BaseModel, name="FNO"):
    """
    N维傅里叶神经算子（Fourier Neural Operator, FNO）。
    FNO的核心目标是学习**离散在规则网格上的函数空间**之间的映射，
    其核心创新是用**频域的谱卷积**替代传统CNN的空域卷积，能高效捕捉函数的全局特征，
    特别适用于偏微分方程（PDE）、流体力学等连续系统的建模（论文：https://arxiv.org/pdf/2010.08895）。

    核心参数（必选）
    ---------------
    n_modes : Tuple[int, ...]
        傅里叶层中沿每个维度保留的**模态数**（频域特征数），FNO的维度由该元组长度决定（如(12,12)为2维）。
        注意：模态数需 < 最大分辨率//2（奈奎斯特频率），过小会丢失高频特征，过大则增加计算量。
    in_channels : int
        输入函数的通道数（由具体问题决定，如单通道标量场、3通道矢量场）。
    out_channels : int
        输出函数的通道数（由具体问题决定）。
    hidden_channels : int
        FNO的"宽度"（隐藏层通道数），直接决定模型参数量和表达能力。
        新手推荐起始值：64，需更强表达能力时可增至128/256（需同步调整升维/投影通道比例）。

    可选参数（常用）
    ---------------
    n_layers : int, optional
        傅里叶层的数量，默认4层（平衡性能与计算量）。
    lifting_channel_ratio : Number, optional
        升维层通道数与hidden_channels的比例，默认2（升维通道数=2*hidden_channels）。
    projection_channel_ratio : Number, optional
        投影层通道数与hidden_channels的比例，默认2（投影通道数=2*hidden_channels）。
    positional_embedding : Union[str, nn.Module], optional
        输入的位置编码方式（为网格数据添加空间位置信息）：
        - "grid"：默认值，为输入追加[0,1]范围的网格位置编码；
        - GridEmbeddingND/2D：直接使用自定义的位置编码模块；
        - None：不使用位置编码。
    non_linearity : nn.Module, optional
        非线性激活函数，默认F.gelu（高斯误差线性单元，比ReLU更稳定）。
    domain_padding : Union[Number, List[Number]], optional
        域填充百分比（缓解谱卷积的边界效应）：
        - None：不填充；
        - 数值/列表：按比例为各维度填充边界。
    complex_data : bool, optional
        是否处理复数数据（如电磁场、波函数），默认False。

    高级参数（进阶调优）
    ---------------
    norm : Literal["ada_in", "group_norm", "instance_norm"], optional
        归一化层类型，默认None（简单场景无需归一化）。
    use_channel_mlp : bool, optional
        是否在每个FNO块后添加通道混合MLP，默认True（增强通道交互）。
    fno_skip / channel_mlp_skip : Literal["linear", "identity", "soft-gating", None], optional
        残差连接类型，分别对应FNO层/通道MLP的残差策略，默认"linear"/"soft-gating"。
    factorization : str, optional
        张量分解方式（用于TFNO），可选"Tucker"/"CP"/"TT"，默认None（稠密FNO）。
    rank : float, optional
        张量分解的秩（分解后参数量=rank*原参数量），默认1.0（稠密），TFNO建议设为0.1~0.5。
    fno_block_precision : str, optional
        谱卷积精度模式，默认"full"（全精度），"mixed"（混合精度）可加速训练。
    """

    def __init__(
        self,
        n_modes: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        lifting_channel_ratio: Number = 2,
        projection_channel_ratio: Number = 2,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity: nn.Module = F.gelu,
        norm: Literal["ada_in", "group_norm", "instance_norm"] = None,
        complex_data: bool = False,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal["linear", "identity", "soft-gating", None] = "soft-gating",
        fno_skip: Literal["linear", "identity", "soft-gating", None] = "linear",
        resolution_scaling_factor: Union[Number, List[Number]] = None,
        domain_padding: Union[Number, List[Number]] = None,
        fno_block_precision: str = "full",
        stabilizer: str = None,
        max_n_modes: Tuple[int, ...] = None,
        factorization: str = None,
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: str = "factorized",
        decomposition_kwargs: dict = None,
        separable: bool = False,
        preactivation: bool = False,
        conv_module: nn.Module = SpectralConv,
    ):
        # 初始化分解参数默认值（未传入则为空字典）
        if decomposition_kwargs is None:
            decomposition_kwargs = {}
        # 调用父类BaseModel的初始化方法
        super().__init__()
        
        # 核心属性初始化：从n_modes长度推断FNO维度（如(12,12)→2维）
        self.n_dim = len(n_modes)
        self._n_modes = n_modes  # 私有变量：存储模态数（通过@property暴露）
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers

        # 计算升维/投影层通道数（按比例）
        self.lifting_channel_ratio = lifting_channel_ratio
        self.lifting_channels = int(lifting_channel_ratio * hidden_channels)
        self.projection_channel_ratio = projection_channel_ratio
        self.projection_channels = int(projection_channel_ratio * hidden_channels)

        # 保存激活函数、张量分解、残差连接等核心配置
        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fno_skip = (fno_skip,)
        self.channel_mlp_skip = (channel_mlp_skip,)
        self.complex_data = complex_data
        self.fno_block_precision = fno_block_precision

        ## 1. 初始化位置编码层
        if positional_embedding == "grid":
            # 默认网格编码：各维度边界为[0,1]（适配常规网格数据）
            spatial_grid_boundaries = [[0.0, 1.0]] * self.n_dim
            self.positional_embedding = GridEmbeddingND(
                in_channels=in_channels,
                dim=self.n_dim,
                grid_boundaries=spatial_grid_boundaries,
            )
        elif isinstance(positional_embedding, GridEmbedding2D):
            # 2维编码需匹配FNO维度
            if self.n_dim == 2:
                self.positional_embedding = positional_embedding
            else:
                raise ValueError(f"错误：{self.n_dim}维FNO无法使用2维GridEmbedding2D")
        elif isinstance(positional_embedding, GridEmbeddingND):
            # 直接使用自定义N维编码
            self.positional_embedding = positional_embedding
        elif positional_embedding is None:
            # 禁用位置编码
            self.positional_embedding = None
        else:
            raise ValueError(f"不支持的位置编码类型：{positional_embedding}（仅支持'grid'/GridEmbeddingND/None）")

        ## 2. 初始化域填充层（仅当填充值>0时生效）
        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                resolution_scaling_factor=resolution_scaling_factor,
            )
        else:
            self.domain_padding = None

        ## 3. 处理分辨率缩放因子（统一为列表格式）
        if resolution_scaling_factor is not None:
            if isinstance(resolution_scaling_factor, (float, int)):
                # 单个值→所有层使用相同缩放
                resolution_scaling_factor = [resolution_scaling_factor] * self.n_layers
        self.resolution_scaling_factor = resolution_scaling_factor

        ## 4. 初始化FNO核心块（多层谱卷积+残差+激活）
        self.fno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            resolution_scaling_factor=resolution_scaling_factor,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            norm=norm,
            preactivation=preactivation,
            fno_skip=fno_skip,
            channel_mlp_skip=channel_mlp_skip,
            complex_data=complex_data,
            max_n_modes=max_n_modes,
            fno_block_precision=fno_block_precision,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            conv_module=conv_module,
            n_layers=n_layers,
        )

        ## 5. 初始化升维层（将输入映射到高维隐空间）
        # 升维层输入通道数 = 原始输入通道数 + 位置编码通道数（若有）
        lifting_in_channels = in_channels
        if self.positional_embedding is not None:
            lifting_in_channels += self.n_dim  # 每个维度添加1个位置通道

        # 根据升维通道数选择MLP结构（2层/1层）
        if self.lifting_channels:
            self.lifting = ChannelMLP(
                in_channels=lifting_in_channels,
                out_channels=hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim,
                non_linearity=non_linearity,
            )
        else:
            # 无升维通道→线性层（1层MLP）
            self.lifting = ChannelMLP(
                in_channels=lifting_in_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                n_layers=1,
                n_dim=self.n_dim,
                non_linearity=non_linearity,
            )
        # 复数数据适配
        if self.complex_data:
            self.lifting = ComplexValued(self.lifting)

        ## 6. 初始化投影层（将高维隐空间映射到输出通道）
        self.projection = ChannelMLP(
            in_channels=hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )
        # 复数数据适配
        if self.complex_data:
            self.projection = ComplexValued(self.projection)

    def forward(self, x, output_shape=None, **kwargs):
        """
        FNO前向传播核心逻辑（输入→位置编码→升维→填充→FNO块→去填充→投影→输出）

        参数
        ----------
        x : torch.Tensor
            输入张量，形状为 [batch_size, in_channels, dim1, dim2, ..., dimN]
        output_shape : tuple/list/None
            为奇数形状输入指定输出形状（解决维度对齐问题）
        **kwargs : 
            兼容额外参数（会被忽略并警告）
        """
        # 处理未预期的关键字参数
        if kwargs:
            warnings.warn(
                f"FNO.forward()收到未预期参数：{list(kwargs.keys())}，已忽略",
                UserWarning,
                stacklevel=2,
            )

        # 统一output_shape为列表（每个FNO块对应一个形状）
        if output_shape is None:
            output_shape = [None] * self.n_layers
        elif isinstance(output_shape, tuple):
            # 仅指定最后一层输出形状
            output_shape = [None] * (self.n_layers - 1) + [output_shape]

        # 步骤1：添加位置编码
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        # 步骤2：升维（输入→高维隐空间）
        x = self.lifting(x)

        # 步骤3：域填充（缓解边界效应）
        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        # 步骤4：依次通过所有FNO块
        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        # 步骤5：移除域填充
        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        # 步骤6：投影（高维隐空间→输出通道）
        x = self.projection(x)

        return x

    @property
    def n_modes(self):
        """n_modes属性的读取方法（封装私有变量）"""
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        """
        n_modes属性的修改方法（同步更新FNO块中的模态数）
        确保模态数的修改能生效到核心的谱卷积层
        """
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes


def partialclass(new_name, cls, *args, **kwargs):
    """
    工具函数：动态创建新类，继承指定类并修改__init__默认参数
    解决functools.partial创建的类无法继承、无类名的问题

    参数
    ----------
    new_name : str
        新类名称
    cls : class
        父类
    *args/**kwargs : 
        父类__init__的默认参数

    返回
    ----------
    class
        新创建的子类
    """
    # 创建偏方法：修改父类__init__默认参数
    __init__ = partialmethod(cls.__init__, *args, **kwargs)
    # 动态创建类：继承父类、保留文档和forward方法
    return type(
        new_name,
        (cls,),
        {
            "__init__": __init__,
            "__doc__": cls.__doc__,
            "forward": cls.forward,
        },
    )


class TFNO(FNO):
    """
    塔克张量分解傅里叶神经算子（TFNO）
    对FNO的权重进行Tucker张量分解，参数量仅为稠密FNO的10%~50%（由rank控制），
    大幅降低计算成本同时保持表达能力。

    核心默认参数
    ---------------
    factorization : str = "Tucker"（塔克分解）
    rank : float = 0.1（分解后参数量=10%稠密FNO）

    其他参数完全继承自FNO类
    """

    def __init__(self, *args, **kwargs):
        # 设置默认分解参数（未传入时生效）
        kwargs.setdefault("factorization", "Tucker")
        kwargs.setdefault("rank", 0.1)
        # 调用FNO父类初始化
        super().__init__(*args, **kwargs)