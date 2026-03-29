"""Train and evaluate a Push-T imitation policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
import wandb
from torch import nn
from torch.utils.data import DataLoader

from hw1_imitation.data import (
    Normalizer,
    PushtChunkDataset,
    download_pusht,
    load_pusht_zarr,
)
from hw1_imitation.model import build_policy, PolicyType
from hw1_imitation.evaluation import Logger, evaluate_policy

LOGDIR_PREFIX = "exp"

#->定义一个数据类，用于存储训练配置，比如数据集路径、模型类型、训练参数等
@dataclass
class TrainConfig:
    # 数据集路径
    data_dir: Path = Path("data")

    # 策略类型，MSE或Flow
    policy_type: PolicyType = "mse"
    # 去噪步数，用于Flow策略（MSE策略无影响）
    flow_num_steps: int = 10
    # 动作块大小
    chunk_size: int = 8

    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.0
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    # 训练轮数
    num_epochs: int = 300
    # 评估频率，以训练步数为单位
    eval_interval: int = 10_000
    num_video_episodes: int = 5
    video_size: tuple[int, int] = (256, 256)
    # 训练日志频率，以训练步数为单位
    log_interval: int = 100
    # 随机种子
    seed: int = 42
    # WandB项目名称
    wandb_project: str = "hw1-imitation"
    # 实验名称后缀，用于日志和WandB
    exp_name: str | None = None

#->解析训练配置，意义：解析命令行参数，并返回一个TrainConfig对象
def parse_train_config(
    args: list[str] | None = None,
    *,
    defaults: TrainConfig | None = None,
    description: str = "Train a Push-T MLP policy.",
) -> TrainConfig:#->函数返回值是一个TrainConfig对象
    defaults = defaults or TrainConfig()#如果defaults为None，则使用默认配置
    return tyro.cli(
        TrainConfig,    
        args=args,
        default=defaults,
        description=description,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed) #设置numpy的随机种子
    torch.manual_seed(seed) #设置torch的随机种子
    torch.cuda.manual_seed_all(seed) #设置cuda的随机种子

#->将训练配置转换为字典，意义：将TrainConfig对象转换为字典，以便于WandB记录配置，方便后续的实验结果分析
def config_to_dict(config: TrainConfig) -> dict[str, Any]: 
    data = asdict(config) #将config对象转换为字典
    for key, value in data.items():
        if isinstance(value, Path): 
            data[key] = str(value) #将Path对象转换为字符串,因为wandb只接受字符串类型的配置
    return data

#->运行训练，意义：设置随机种子，选择设备，打印设备信息，读取数据，构建模型，初始化日志，优化器，主训练循环
def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) 读取/下载数据，并构造标准化器（训练时 state/action 会被标准化）
    zarr_path = download_pusht(config.data_dir)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)

    # 2) Dataset: (obs, action_chunk) 监督学习样本对
    #    obs: (state_dim,), chunk: (chunk_size, action_dim)
    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        normalizer=normalizer,
    )

    # 3) DataLoader: 自动把多个样本拼成 batch
    #    state: (B, state_dim), action_chunk: (B, chunk_size, action_dim)
    #batch size（B: 批量大小,意义：每个批次包含的样本数量，用于控制训练过程中的计算量和内存使用）
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # 4) 构建模型（MSE or Flow），并放到 device
    model = build_policy(
        config.policy_type,
        state_dim=states.shape[1],#state_dim: 状态维度，即观测维度，Push-T 通常为 5,[1]表示第二维，即状态维度。
        action_dim=actions.shape[1],
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
    ).to(device)

    # 5) 日志目录 + WandB 初始化 + 本地 logger（写 log.csv + 拷贝 wandb 目录用于提交）
    exp_name = f"seed_{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.exp_name is not None:
        exp_name += f"_{config.exp_name}"
    log_dir = Path(LOGDIR_PREFIX) / exp_name
    wandb.init(
        project=config.wandb_project, config=config_to_dict(config), name=exp_name
    )
    logger = Logger(log_dir)

    # 6) 优化器（AdamW 支持 weight_decay；若你更想用 Adam 也可以）
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # 7) 主训练循环
    # 训练步数：以“优化器更新次数”为单位（每个 batch 1 step）
    step = 0
    model.train() 

    for epoch in range(config.num_epochs):
        for batch_state, batch_action_chunk in loader:
            # batch_state: (B, state_dim)
            # batch_action_chunk: (B, chunk_size, action_dim)
            batch_state = batch_state.to(device)
            batch_action_chunk = batch_action_chunk.to(device)

            loss = model.compute_loss(batch_state, batch_action_chunk)  # scalar tensor ()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1

            # --- 训练日志（更高频）---
            if step % config.log_interval == 0:
                logger.log(
                    {
                        "train/loss": float(loss.detach().cpu().item()),
                        "train/epoch": float(epoch),
                    },
                    step=step,
                )

            # --- 评测日志（必须定期调用，用于 grading + 视频）---
            if step % config.eval_interval == 0:
                evaluate_policy(
                    model=model,
                    normalizer=normalizer,
                    device=device,
                    chunk_size=config.chunk_size,
                    video_size=config.video_size,
                    num_video_episodes=config.num_video_episodes,
                    flow_num_steps=config.flow_num_steps,
                    step=step,
                    logger=logger,
                )
                model.train()

    logger.dump_for_grading()


def main() -> None:
    config = parse_train_config()
    run_training(config)


if __name__ == "__main__":
    main()
