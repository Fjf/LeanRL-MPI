import logging
import os
import random
import time
from collections import deque
from functools import partial
from typing import Sequence, Callable, Optional, Union, List, Tuple, Any

import gymnasium as gym
import mpi4py
import numpy as np
import torch
import tqdm
import wandb
from mpi4py import MPI
from numpy._typing import NDArray
from torch import optim as optim, nn

from leanrl.mpi.env_wrappers import RPCEnvWrapper, MPIFunctionParams
from leanrl.mpi.network import Agent
from leanrl.mpi.utils import make_env, Args


# ALGO Logic: Storage setup
class Storage:
    def __init__(self, num_steps, num_envs, single_observation_space, single_action_space, device):
        self.obs = torch.zeros((num_steps, num_envs) + single_observation_space.shape).to(device)
        self.actions = torch.zeros((num_steps, num_envs) + single_action_space.shape).to(device)
        self.logprobs = torch.zeros((num_steps, num_envs)).to(device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(device)
        self.dones = torch.zeros((num_steps, num_envs)).to(device)
        self.values = torch.zeros((num_steps, num_envs)).to(device)

    def add(self, next_obs, next_done, value, action, logprob, reward, step):
        self.obs[step] = next_obs
        self.actions[step] = action
        self.logprobs[step] = logprob
        self.rewards[step] = reward
        self.dones[step] = next_done
        self.values[step] = value

    def bootstrap(self, agent, args, device, next_done, next_obs):
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + args.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values
        return advantages, returns


def main(args: Args):
    wandb.init(
        project="ppo_continuous_action",
        name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{args.run_name}",
        config=vars(args),
        save_code=True,
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    logging.error("Creating envs")
    envs = MPIVectorEnv(
        [
            make_env(
                env_id=args.env_id,
                idx=worker_id,
                capture_video=args.capture_video,
                run_name=args.run_name,
                gamma=args.gamma,
                extra_wrapper_fns=[partial(RPCEnvWrapper, MPI.COMM_WORLD, worker_id)],
            )
            for worker_id in range(1, mpi4py.MPI.COMM_WORLD.Get_size())
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    logging.error("Creating agent")
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    logging.error("Creating storage")

    storage = Storage(args.num_steps, args.num_envs, envs.single_observation_space, envs.single_action_space, device)
    avg_returns = deque(maxlen=20)

    logging.error("Staring game")

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    max_ep_ret = -float("inf")
    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))
    global_step_burnin = None
    start_time = None
    desc = ""

    # profiler = Profiler()

    for iteration in pbar:
        if iteration == args.measure_burnin:
            global_step_burnin = global_step
            start_time = time.time()

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            reward = torch.tensor(reward).to(device).view(-1)

            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            storage.add(next_obs, next_done, value.flatten(), action, logprob, reward, step)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    r = float(info["episode"]["r"].reshape(()))
                    max_ep_ret = max(max_ep_ret, r)
                    avg_returns.append(r)
                desc = f"global_step={global_step}, episodic_return={torch.tensor(avg_returns).mean(): 4.2f} (max={max_ep_ret: 4.2f})"

        # bootstrap value if not done TODO: do this for MPI buffer
        advantages, returns = storage.bootstrap(agent, args, device, next_done, next_obs)

        # flatten the batch
        b_obs = storage.obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = storage.logprobs.reshape(-1)
        b_actions = storage.actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = storage.values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if global_step_burnin is not None and iteration % 10 == 0:
            speed = (global_step - global_step_burnin) / (time.time() - start_time)
            pbar.set_description(f"speed: {speed: 4.1f} sps, " + desc)
            with torch.no_grad():
                logs = {
                    "episode_return": np.array(avg_returns).mean(),
                    "logprobs": b_logprobs.mean(),
                    "advantages": advantages.mean(),
                    "returns": returns.mean(),
                    "values": storage.values.mean(),
                    "gn": gn,
                }
            wandb.log(
                {
                    "speed": speed,
                    **logs,
                },
                step=global_step,
            )

    logging.info("Closing envs")
    envs.close()

    logging.info("Killing workers")
    envs.mpi_kill()


class MPIVectorEnv(gym.vector.VectorEnv):
    def __init__(self, env_fns: Sequence[Callable[[], RPCEnvWrapper]]):
        self.envs: List[RPCEnvWrapper] = [fn() for fn in env_fns]

        observation_space = self.envs[0].observation_space
        action_space = self.envs[0].action_space

        super().__init__(len(env_fns), observation_space, action_space)

    def mpi_kill(self):
        for worker_id in range(1, MPI.COMM_WORLD.Get_size()):
            MPI.COMM_WORLD.send(MPIFunctionParams(source=0, fn="kill", args=None, kwargs=None).to_obj(), dest=worker_id)

    def _stack_data(self, data: List[Tuple[Any]]):
        vec_response = list(zip(*data))
        stacked_response = []
        for response in vec_response:
            if isinstance(response[0], np.ndarray):
                stacked_response.append(np.stack(response))
            else:
                stacked_response.append(response)
        return tuple(stacked_response)

    def reset(self, *, seed: Optional[Union[int, List[int]]] = None, options: Optional[dict] = None):
        delayed = []
        for idx, env in enumerate(self.envs):
            if isinstance(seed, int):
                delayed.append(env.reset(seed=seed, options=options))
            else:
                delayed.append(env.reset(seed=seed[idx], options=options))

        raw_data = [d() for d in delayed]
        data = self._stack_data(raw_data)
        return *data,

    def step(self, actions) -> Tuple[Any, NDArray[Any], NDArray[Any], NDArray[Any], dict]:
        delayed = []
        for action, env in zip(actions, self.envs):
            delayed.append(env.step(action))

        return self._stack_data([d() for d in delayed])

    def call(self, name: str, *args, **kwargs) -> List[Any]:
        delayed = []
        for env in self.envs:
            method = getattr(env, name)
            delayed.append(method(*args, **kwargs))

        return [d() for d in delayed]

    def get_attr(self, name: str):
        return super().get_attr(name)

    def set_attr(self, name: str, values: Union[list, tuple, object]):
        super().set_attr(name, values)

    def close_extras(self, **kwargs):
        super().close_extras(**kwargs)

    def close(self, **kwargs):
        super().close(**kwargs)
