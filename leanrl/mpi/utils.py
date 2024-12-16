import datetime
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Callable

import gymnasium
import gymnasium as gym
import numpy as np
import tyro
from mpi4py import MPI


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 10000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 1024
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    # use_target_kl: bool = False
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed at runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed at runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed at runtime)"""
    run_name: str = ""
    """name of the run for wandb (computed at runtime)"""

    measure_burnin: int = 3

    """ ==== MPI ARGS ==== """
    num_comm_tree_chunks: int = 8
    """the amount of chunks per tree-split of the mpi worker groups"""


def parse_args(rank) -> Args:
    args = tyro.cli(Args, console_outputs=False if rank != 0 else True)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    args.num_envs = MPI.COMM_WORLD.Get_size() - 1

    args.run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"
    return args


def make_env(env_id, idx, capture_video, run_name, gamma, extra_wrapper_fns: List[Callable[[gymnasium.Env], gymnasium.Wrapper]] = None):
    if extra_wrapper_fns is None:
        extra_wrapper_fns = []

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        for wrapper_fn in extra_wrapper_fns:
            env = wrapper_fn(env)
        return env

    return thunk


class Profiler:
    def __init__(self):
        self.segment_starts: dict[str, datetime] = {}
        self.segments: dict[str, float] = defaultdict(float)
        self.active_segment: Optional[str] = None

    def start(self, segment_name: str):
        if self.active_segment is not None:
            self.stop(self.active_segment)

        self.segment_starts[segment_name] = datetime.datetime.now()
        self.active_segment = segment_name

    def stop(self, segment_name):
        if segment_name not in self.segment_starts:
            return

        start_time = self.segment_starts[segment_name]
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        self.segments[segment_name] += duration

        del self.segment_starts[segment_name]
        if self.active_segment == segment_name:
            self.active_segment = None

    def get_segment_time(self, segment_name):
        return self.segments.get(segment_name, 0.0)

    def print(self):
        logging.info("Profiling information:")
        for segment, time in self.segments.items():
            print(f"\t{segment}: {time} seconds")
