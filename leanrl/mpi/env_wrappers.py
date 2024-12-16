import logging
from functools import partial
from typing import Optional, Dict, Any

import gymnasium
import mpi4py
import numpy as np
import torch
from gymnasium import Env


class RPCEnvWrapperWorker(Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prev_rewards = np.zeros(0)
        # self.buffer = MPIRolloutBuffer(
        #     buffer_size=self.buffer_size, observation_space=self.observation_space,
        #     action_space=self.action_space, n_workers=self.num_envs, gamma=gamma, gae_lambda=gae_lambda
        # )
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

    def mpi_kill(self):
        """
        This function is only here to allow for global termination of the task.
        :return:
        """
        ...


class MPIFunctionParams:
    def __init__(self, source, fn, args, kwargs):
        self.source = source
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def to_obj(self):
        return (self.source, self.fn, self.args, self.kwargs)


class RPCEnvWrapper(gymnasium.Wrapper):
    def __init__(self, comm, remote, env: RPCEnvWrapperWorker):  # noqa
        super().__init__(env)
        self.comm: mpi4py.MPI.Comm = comm
        self.remote = remote

        self.method_lookup: dict[str, int] = {name: number + 1 for number, name in zip(range(len(dir(env))), dir(env))}
        self.observation_buffer = np.zeros(env.observation_space.shape)

        for method_name in dir(env):
            if method_name.startswith("_"):
                continue

            prop = getattr(env, method_name)
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
            params = MPIFunctionParams(0, method.__name__, converter(args), converter(kwargs))
            if method.__name__ == "step":
                np_array: np.ndarray = params.args[0]
                self.comm.Send(np_array.data, self.remote, tag=self.method_lookup["step"])

                def recv_func(source):
                    self.comm.Recv(self.observation_buffer.data, source=source)
                    responses = self.comm.recv(source=source)
                    return self.observation_buffer, *responses


            else:
                self.comm.send(params.to_obj(), self.remote, tag=self.method_lookup[method.__name__])
                recv_func = self.comm.recv


            func = partial(recv_func, source=self.remote)
            if not delayed:
                return func()

            return func

        return _wrapper