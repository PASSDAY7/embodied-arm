"""网络结构定义模块。

本模块以"策略网络注册表 + 工厂函数"的方式组织，统一对外暴露
``build_network(config)``，通过配置项 ``model.name`` 即可切换 MLP / CNN / GRU，
无需修改训练脚本。

所有网络约定：
* 输入: 状态张量 ``(batch, state_dim)``
* 输出: 动作张量 ``(batch, action_dim)``；当 ``model.chunk_size = K > 1`` 时，
  输出为**动作块** ``(batch, K, action_dim)``（W4 动作块行为克隆 Action Chunking）
"""

from __future__ import annotations

from typing import Dict, List, Type

import torch
import torch.nn as nn


# =============================================================================
# 1. 策略网络基类
# =============================================================================
class BasePolicy(nn.Module):
    """策略网络基类：统一处理动作缩放、输出压缩与动作块整形。

    Args:
        action_dim: 输出动作维度。
        action_scale: 动作缩放系数，对应动作空间 ``[-action_scale, action_scale]``。
        squash: 为 ``True`` 时使用 Tanh 将输出压缩到 [-1, 1] 再缩放。
        chunk_size: 动作块长度 ``K``（W4 动作块行为克隆）。

            * ``K > 1``：一次前向预测未来 ``K`` 步动作，输出头维度为
              ``K * action_dim``，最终整形成 ``(batch, K, action_dim)``；
            * ``K == 1``：退化为单步 BC，输出 ``(batch, action_dim)``，
              与 W2/W3 版本**逐位一致**（保持向后兼容）。
    """

    def __init__(
        self,
        action_dim: int = 1,
        action_scale: float = 2.0,
        squash: bool = True,
        chunk_size: int = 1,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_scale = float(action_scale)
        self.squash = squash
        self.chunk_size = max(1, int(chunk_size))
        # 输出层维度：K * action_dim（动作块一次性预测，而非单步动作）
        self.output_dim = self.chunk_size * self.action_dim

    def _scale_output(self, x: torch.Tensor) -> torch.Tensor:
        """将网络原始输出映射到动作空间。"""
        if self.squash:
            x = torch.tanh(x)
        return x * self.action_scale

    def _format_output(self, x: torch.Tensor) -> torch.Tensor:
        """把输出头结果整形为动作块维度 ``(batch, K, action_dim)``。"""
        if self.chunk_size == 1:
            return x                                        # (batch, action_dim) 单步 BC
        return x.reshape(-1, self.chunk_size, self.action_dim)

    def forward_actions(self, x: torch.Tensor) -> torch.Tensor:
        """统一推理接口：无论 ``K`` 取值，均返回 ``(batch, K, action_dim)``。

        推理侧（闭环 rollout）只依赖本方法，从而单步策略（K=1）与动作块策略
        可以共用同一套执行逻辑。
        """
        out = self.forward(x)
        return out if out.dim() == 3 else out.unsqueeze(1)


# =============================================================================
# 2. MLP 策略网络
# =============================================================================
class MLPPolicy(BasePolicy):
    """多层感知机策略网络：适用于低维状态输入（默认）。"""

    def __init__(
        self,
        state_dim: int = 3,
        action_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        action_scale: float = 2.0,
        squash: bool = True,
        activation: str = "relu",
        chunk_size: int = 1,
    ) -> None:
        super().__init__(
            action_dim=action_dim, action_scale=action_scale, squash=squash, chunk_size=chunk_size
        )

        act_layer = _resolve_activation(activation)
        layers: List[nn.Module] = []
        in_dim = state_dim
        for _ in range(max(1, num_layers)):
            layers += [nn.Linear(in_dim, hidden_dim), act_layer()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.output_dim))    # K * action_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._format_output(self._scale_output(self.net(x)))


# =============================================================================
# 3. CNN 策略网络（1D 卷积）
# =============================================================================
class CNNPolicy(BasePolicy):
    """一维卷积策略网络：把状态向量视为长度为 state_dim 的单通道序列。

    适合状态维度较高、存在局部相关性的场景；本项目主要用于演示
    ``model.name: cnn`` 的切换能力。
    """

    def __init__(
        self,
        state_dim: int = 3,
        action_dim: int = 1,
        conv_channels: List[int] | None = None,
        kernel_size: int = 3,
        hidden_dim: int = 64,
        action_scale: float = 2.0,
        squash: bool = True,
        chunk_size: int = 1,
    ) -> None:
        super().__init__(
            action_dim=action_dim, action_scale=action_scale, squash=squash, chunk_size=chunk_size
        )
        conv_channels = conv_channels or [32, 64]
        padding = kernel_size // 2

        conv_layers: List[nn.Module] = []
        in_ch = 1
        for out_ch in conv_channels:
            conv_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
            ]
            in_ch = out_ch
        conv_layers.append(nn.AdaptiveAvgPool1d(1))
        self.features = nn.Sequential(*conv_layers)

        self.head = nn.Sequential(
            nn.Linear(in_ch, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),           # K * action_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, state_dim) -> (batch, 1, state_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.features(x).flatten(1)
        return self._format_output(self._scale_output(self.head(feat)))


# =============================================================================
# 4. GRU 策略网络
# =============================================================================
class GRUPolicy(BasePolicy):
    """GRU 策略网络：将状态向量视为长度 state_dim、特征维度 1 的序列。

    取最后时间步的隐藏状态做动作预测，可捕捉输入分量间的时序依赖。
    """

    def __init__(
        self,
        state_dim: int = 3,
        action_dim: int = 1,
        gru_hidden: int = 64,
        gru_layers: int = 1,
        hidden_dim: int = 64,
        action_scale: float = 2.0,
        squash: bool = True,
        chunk_size: int = 1,
    ) -> None:
        super().__init__(
            action_dim=action_dim, action_scale=action_scale, squash=squash, chunk_size=chunk_size
        )
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),           # K * action_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, state_dim) -> (batch, state_dim, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        return self._format_output(self._scale_output(self.head(out[:, -1, :])))


# =============================================================================
# 5. 注册表与工厂函数
# =============================================================================
NETWORKS: Dict[str, Type[BasePolicy]] = {
    "mlp": MLPPolicy,
    "cnn": CNNPolicy,
    "gru": GRUPolicy,
}


def _resolve_activation(name: str) -> Type[nn.Module]:
    """按名称解析激活函数层。"""
    mapping: Dict[str, Type[nn.Module]] = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"未知激活函数 '{name}'，可选: {list(mapping)}")
    return mapping[key]


def build_network(config: Dict) -> BasePolicy:
    """根据配置字典构建策略网络。

    Args:
        config: 完整实验配置（含 ``model`` 段），或仅 ``model`` 段本身。

    Returns:
        实例化的策略网络（尚未移动到设备）。
    """
    model_cfg = config.get("model", config)
    name = str(model_cfg.get("name", "mlp")).lower()
    if name not in NETWORKS:
        raise ValueError(f"未知网络类型 '{name}'，可选: {list(NETWORKS)}")

    shared = dict(
        state_dim=int(model_cfg.get("state_dim", 3)),
        action_dim=int(model_cfg.get("action_dim", 1)),
        action_scale=float(model_cfg.get("action_scale", 2.0)),
        squash=bool(model_cfg.get("squash", True)),
        chunk_size=int(model_cfg.get("chunk_size", 1)),      # W4: 动作块长度 K（1 = 单步 BC）
    )

    if name == "mlp":
        return MLPPolicy(
            **shared,
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            activation=str(model_cfg.get("activation", "relu")),
        )
    if name == "cnn":
        return CNNPolicy(
            **shared,
            conv_channels=list(model_cfg.get("conv_channels", [32, 64])),
            kernel_size=int(model_cfg.get("kernel_size", 3)),
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        )
    # gru
    return GRUPolicy(
        **shared,
        gru_hidden=int(model_cfg.get("gru_hidden", 64)),
        gru_layers=int(model_cfg.get("gru_layers", 1)),
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
    )
