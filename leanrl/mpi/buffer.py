import os
from collections import defaultdict
from functools import partial
from typing import Generator

import mpi4py.MPI
import numpy as np
import torch
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from tqdm import tqdm


def to_torch(array: np.ndarray, device: str = "cpu", copy: bool = True) -> th.Tensor:
    """
    Convert a numpy array to a PyTorch tensor.
    Note: it copies the data by default

    :param array:
    :param copy: Whether to copy or not the data (may be useful to avoid changing things
        by reference). This argument is inoperative if the device is not the CPU.
    :return:
    """
    if copy:
        return th.tensor(array, device=device)
    return th.as_tensor(array, device=device)


class MPIRolloutBuffer:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_workers: int = 1,
    ):
        self.n_workers = n_workers
        self.buffer_size = buffer_size
        self.pos = 0

        self.action_space = action_space
        self.observation_space = observation_space
        self.action_dim = get_action_dim(action_space)
        self.obs_shape = get_obs_shape(observation_space)

        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_workers, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_workers, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_workers), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_workers), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_workers), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_workers), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_workers), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_workers), dtype=np.float32)
        self.generator_ready = False
        self.pos = 0
        self.full = False

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            value: th.Tensor,
            log_prob: th.Tensor,
    ) -> None:
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape(self.obs_shape)

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape(self.action_dim)

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size=1, output_type: str = "torch") -> Generator[RolloutBufferSamples, None, None]:
        if output_type == "torch":
            map_fn = partial(to_torch, device="cpu")
        elif output_type == "numpy":
            map_fn = np.array
        else:
            raise ValueError(f"Invalid `output_type(={output_type})` passed.")

        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_workers)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = BaseBuffer.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create mini-batches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_workers

        start_idx = 0
        while start_idx < self.buffer_size * self.n_workers:
            yield self._get_samples(indices[start_idx: start_idx + batch_size], map_fn=map_fn)
            start_idx += batch_size

        # for i in range(self.buffer_size):
        #     yield self._get_samples(indices[i])

    def _get_samples(
            self,
            batch_inds: np.ndarray,
            map_fn=np.array,
    ) -> RolloutBufferSamples:
        data = (
            self.observations.view()[batch_inds],
            self.actions.view()[batch_inds],
            self.values.view()[batch_inds].flatten(),
            self.log_probs.view()[batch_inds].flatten(),
            self.advantages.view()[batch_inds].flatten(),
            self.returns.view()[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(map_fn, data)))

    def emit_batches(self, comm: mpi4py.MPI.Comm):
        # Send ping to request batch_size from mater
        comm.send([0], dest=0)
        batch_size = comm.recv(source=0)

        for sample in self.get(batch_size=batch_size, output_type="numpy"):
            ack = comm.recv(source=0)  # Recv request to send data
            comm.send(sample, dest=0)  # Send data
            del sample


class WorkerMPIRolloutBuffer:
    returns: np.ndarray
    values: np.ndarray

    def __init__(self, comm, env: MpiRPCVecEnv, prefetch_factor=1):
        self.prefetch_factor = prefetch_factor
        self.n_workers = comm.size - 1
        self.env = env
        self.comm = comm
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def add(self, *args):
        self.env.buffer_add(*args)

    def reset(self):
        self.env.buffer_reset()

    def compute_returns_and_advantage(self, *args, **kwargs):
        rv_list = self.env.buffer_compute_returns_and_advantage(*args, **kwargs)
        self.returns, self.values = map(np.array, zip(*rv_list))

    def get(self, batch_size=32):
        def to_tensor(t):
            return torch.as_tensor(t, device=self.device)

        self.env.buffer_emit_batches()

        # Await all workers ready to receive data
        assert batch_size % self.n_workers == 0
        worker_batch_size = batch_size // self.n_workers
        for worker_id in range(1, self.n_workers + 1):
            ack = self.comm.recv(source=worker_id)
            self.comm.send(worker_batch_size, dest=worker_id)

        # Prefetch data from workers
        for current_worker in range(0, self.n_workers * self.prefetch_factor):
            self.comm.send([0], dest=(current_worker % self.n_workers) + 1)

        n_samples = self.env.num_envs * self.env.ep_length

        for i in tqdm(range(0, n_samples, batch_size)):
            batch_test = defaultdict(list)

            # Request less based on prefetch factor (multiplied by batch count)
            if i < n_samples - (batch_size * self.prefetch_factor):
                for current_worker in range(1, self.n_workers + 1):
                    self.comm.send([0], dest=current_worker)

            # Receive data from all
            for current_worker in range(1, self.n_workers + 1):
                sample = self.comm.recv(source=current_worker)

                # batch.append(sample)
                for sample_idx, chunk in enumerate(sample):
                    batch_test[sample_idx].append(chunk)

            data = (
                to_tensor(np.concatenate(batch_test[x]))
                for x in range(len(batch_test.items()))
            )
            big_batch = RolloutBufferSamples(*data)
            yield big_batch

        # They will all send a 'None' when they are done sending data.
        for worker in range(1, self.n_workers + 1):
            self.comm.recv(source=worker)
