"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""
    # 所有策略模型的统一父类，定义公共配置和接口

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        # state_dim: 观测维度（Push-T 通常 5）
        self.state_dim = state_dim
        # action_dim: 单步动作维度（Push-T 通常 2）
        self.action_dim = action_dim
        # chunk_size: 一次预测多少步动作
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""
        # 约定输入 shape:
        # state: (B, state_dim)
        # action_chunk: (B, chunk_size, action_dim)

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""
        # 返回 shape: (B, chunk_size, action_dim)


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""
    # MSE 策略: 直接“一次前向”预测整段动作 chunk

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128), 
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        # MSE policy: 单次前向直接输出整个 action chunk
        # 输入:  state, shape (B, state_dim)
        # 输出:  action_chunk, shape (B, chunk_size, action_dim)
        out_dim = chunk_size * action_dim

        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:  
            # 逐层堆叠 MLP: Linear -> ReLU
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        # 最后一层输出展平后的 chunk 向量
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning (B, chunk_size, action_dim)."""
        pred = self.net(state)  # (B, chunk_size * action_dim)
        # 把展平输出还原成三维: (B, K, A)
        return pred.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        # pred_chunk / action_chunk 形状一致，逐元素做均方误差
        pred_chunk = self.forward(state)
        return torch.nn.functional.mse_loss(pred_chunk, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # 对 MSE policy 来说，采样就是直接前向预测（不需要 num_steps）
        return self.forward(state)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""
    # TODO: 第二部分作业实现（当前占位）

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        # 输入给速度网络的是:
        # [state, noisy_action_chunk_flat, tau]
        in_dim = state_dim + chunk_size * action_dim + 1
        out_dim = chunk_size * action_dim  # 预测速度场 v_theta 的展平输出

        layers: list[nn.Module] = []
        cur = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(cur, h))
            layers.append(nn.ReLU())
            cur = h
        layers.append(nn.Linear(cur, out_dim))
        self.net = nn.Sequential(*layers)

    def _predict_velocity(
        self, state: torch.Tensor, action_chunk: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        """Predict v_theta(state, action_chunk, tau) with shape (B, K, A)."""
        batch_size = state.shape[0] 
        flat_chunk = action_chunk.view(batch_size, -1)  # (B, K*A)
        inp = torch.cat([state, flat_chunk, tau], dim=-1)  # (B, state_dim + K*A + 1)
        vel = self.net(inp)  # (B, K*A)
        return vel.view(batch_size, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        # Flow matching training target:
        # A_0 ~ N(0, I), tau ~ U(0, 1)
        # A_tau = tau * A + (1 - tau) * A_0
        # target velocity = A - A_0
        batch_size = state.shape[0]
        noise = torch.randn_like(action_chunk)  # A_0
        tau = torch.rand(batch_size, 1, device=state.device, dtype=state.dtype)
        tau_expand = tau.view(batch_size, 1, 1)

        noisy_chunk = tau_expand * action_chunk + (1.0 - tau_expand) * noise
        target_velocity = action_chunk - noise
        pred_velocity = self._predict_velocity(state, noisy_chunk, tau)
        return torch.nn.functional.mse_loss(pred_velocity, target_velocity)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # Euler integration:
        # dA_tau/dtau = v_theta(state, A_tau, tau), tau: 0 -> 1
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        batch_size = state.shape[0]
        dt = 1.0 / float(num_steps)
        chunk = torch.randn(
            batch_size,
            self.chunk_size,
            self.action_dim,
            device=state.device,
            dtype=state.dtype,
        )  # A_0

        for i in range(num_steps):
            tau_value = i / float(num_steps)
            tau = torch.full(
                (batch_size, 1),
                fill_value=tau_value,
                device=state.device,
                dtype=state.dtype,
            )
            velocity = self._predict_velocity(state, chunk, tau)
            chunk = chunk + dt * velocity

        return chunk


PolicyType: TypeAlias = Literal["mse", "flow"]
 # policy_type 只能是这两个字符串之一


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    # 简单工厂函数：根据字符串创建对应策略对象
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
