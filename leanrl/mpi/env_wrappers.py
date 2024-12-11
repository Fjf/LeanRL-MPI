from functools import partial
from typing import Optional, Dict, Any

import mpi4py
import numpy as np
import torch
from gymnasium.experimental.vector import AsyncVectorEnv


class RPCEnvWrapperWorker(AsyncVectorEnv):
    def __init__(self, *args, gamma=0.95, gae_lambda=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        from buffer import MPIRolloutBuffer

        self.buffer_size = self.get_attr("buffer_size", indices=0)[0]
        self.prev_rewards = np.zeros(self.num_envs)
        self.buffer = MPIRolloutBuffer(
            buffer_size=self.buffer_size, observation_space=self.observation_space,
            action_space=self.action_space, n_workers=self.num_envs, gamma=gamma, gae_lambda=gae_lambda
        )
        self.reset_info: Optional[Dict[str, Any]] = {}

    def buffer_compute_returns_and_advantage(self, *args, **kwargs):
        self.buffer.compute_returns_and_advantage(*args, **kwargs)
        return self.buffer.returns, self.buffer.values

    def buffer_reset(self):
        self.buffer.reset()

    def buffer_emit_batches(self):
        self.buffer.emit_batches(mpi4py.MPI.COMM_WORLD)

    def buffer_add(self, observations, actions, rewards, episode_starts, values, log_probs):
        self.buffer.add(observations, actions, self.prev_rewards, episode_starts, values, log_probs)
        self.prev_rewards = rewards

    def env_method(self, method, *args, **kwargs):
        method = getattr(self, method)
        return method(*args, **kwargs)

    def is_wrapped(self, data):
        return self.env_is_wrapped(data)

    def ping(self, data):
        return data

    def get_spaces(self):
        return self.observation_space, self.action_space



class RPCEnvWrapper(RPCEnvWrapperWorker):
    def __init__(self, comm, remote, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.comm: mpi4py.MPI.Comm = comm
        self.remote = remote

        for method_name in dir(self):
            if method_name.startswith("_"):
                continue

            prop = getattr(self, method_name)
            if callable(prop):
                self.__setattr__(method_name, self._rpc_wrapper(prop, delayed=True))

    def _rpc_wrapper(self, method, delayed=False):
        def converter(args):
            """
            Tensors need to be moved to the CPU when sending them over MPI, as we cannot guarantee that the
             recipient has access to a GPU.
            """

            def t_c(arg):
                if type(arg) == torch.Tensor:
                    return arg.to("cpu")
                return arg

            if type(args) == tuple:
                return tuple(t_c(arg) for arg in args)
            if type(args) == dict:
                return {k: t_c(arg) for k, arg in args.items()}

        def _wrapper(*args, buffer: torch.Tensor = None, **kwargs):
            self.comm.send((method.__name__, converter(args), converter(kwargs)), self.remote)

            # We cannot write to a buffer that is not passed by the user, we don't know what the resulting data-size
            #  will be.
            if buffer is not None:
                if not isinstance(buffer, torch.Tensor):
                    raise ValueError(f"Cannot write fetched MPI data to non-tensor type `{type(buffer)}`")

                recv_func = partial(self.comm.Recv, buffer=buffer)
            else:
                recv_func = self.comm.recv

            func = partial(recv_func, source=self.remote)
            if not delayed:
                return func()

            return func

        return _wrapper