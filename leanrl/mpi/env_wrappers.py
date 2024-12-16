from functools import partial
from typing import Optional, Dict, Any

import gymnasium
import mpi4py
import numpy as np
import torch
from gymnasium import Env
from mpi4py import MPI


class RPCEnvWrapperWorker(Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prev_rewards = np.zeros(0)
        self.reset_info: Optional[Dict[str, Any]] = {}

    def env_method(self, method, *args, **kwargs):
        method = getattr(self, method)
        return method(*args, **kwargs)

    def is_wrapped(self, data):
        return self.is_wrapped(data)

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
    def __init__(self, comm, remote, env: RPCEnvWrapperWorker, chunk_size=1):  # noqa
        super().__init__(env)
        self.comm: mpi4py.MPI.Comm = comm
        self.remote = remote
        self.method_lookup: dict[str, int] = {name: number + 1 for number, name in zip(range(len(dir(env))), dir(env))}
        self.observation_buffer = np.zeros((chunk_size, *env.observation_space.shape))
        self.rtt_buffer = np.zeros((chunk_size, 3), dtype=np.float32)

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
                    self.comm.Recv([self.observation_buffer, MPI.FLOAT], source=source)
                    self.comm.Recv([self.rtt_buffer, MPI.FLOAT], source=source)
                    infos = self.comm.recv(source=source)
                    return (
                        self.observation_buffer,
                        self.rtt_buffer[:, 0],
                        self.rtt_buffer[:, 1].astype(np.bool_),
                        self.rtt_buffer[:, 2].astype(np.bool_),
                        infos
                    )

            else:
                self.comm.send(params.to_obj(), self.remote, tag=self.method_lookup[method.__name__])
                recv_func = self.comm.recv

            func = partial(recv_func, source=self.remote)
            if not delayed:
                return func()

            return func

        return _wrapper
