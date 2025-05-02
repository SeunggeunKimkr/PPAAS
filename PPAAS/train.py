import gymnasium as gym
import csv
import numpy as np
import torch
import torch.nn.functional as F
import click
from stable_baselines3 import SAC, HerReplayBuffer, PPO, DDPG
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
import sys
import os
import time
from typing import Callable

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from PPAAS.envs.ngspice_env_goal import ngspice_env_goal
from PPAAS.envs.ngspice_env import ngspice_env_cont
from stable_baselines3.common.utils import set_random_seed
import wandb
from wandb.integration.sb3 import WandbCallback


path = os.getcwd()


class train_callback(BaseCallback):
    def __init__(
        self,
        num_corners,
        env,
        verbose=0,
        obj_style="softmax",
        temp=10.0,
        algo="HER",
        pareto_freq=0,
    ):
        super(train_callback, self).__init__(verbose)
        self.tt_sim = 0
        self.full_sim = 0
        self.total_sim = 0
        self.num_corners = num_corners
        self.single_to_multi = False  # transition from single to multi goal
        self.obs, _ = env.reset()
        self.SoF = env.SoF
        self.obj_style = obj_style
        self.temp = temp
        self.algo = algo
        self.pareto_freq = pareto_freq

    def _on_step(self) -> bool:
        if self.SoF:
            if self.locals["infos"][0].get("tt_done"):
                self.total_sim += self.num_corners
                self.full_sim += 1
            else:
                self.total_sim += 1
                self.tt_sim += 1
        else:
            self.total_sim += self.num_corners
            self.full_sim += 1
        ep_corner_norm_std = self.locals["infos"][0].get("ep_corner_norm_std", 1.0)
        wandb.log(
            {
                "tt_sim": self.tt_sim,
                "full_sim": self.full_sim,
                "total_sim": self.total_sim,
                "ep_corner_norm_std": ep_corner_norm_std,
            }
        )

        # Pareto Dominant Goal Sampling
        if self.pareto_freq > 0:
            if self.locals["infos"][0].get("done") or self.locals["infos"][0].get(
                "truncated"
            ):
                obs = self.obs
                obs_tensor_dict = {}
                goals_norm = self.training_env.env_method("get_goals_norm")[0]
                goals_norm_tensor = torch.tensor(
                    goals_norm, dtype=torch.float32, device="cpu"
                )
                num_goals = goals_norm_tensor.shape[0]
                done_padding = torch.ones(num_goals, 2, device="cpu")
                goals_norm_tensor = torch.cat([goals_norm_tensor, done_padding], dim=1)

                for key, value in obs.items():
                    value_tensor = torch.tensor(
                        value, dtype=torch.float32, device="cpu"
                    ).unsqueeze(0)
                    value_tensor = value_tensor.repeat(num_goals, 1)
                    obs_tensor_dict[key] = value_tensor
                obs_tensor_dict["desired_goal"] = goals_norm_tensor

                with torch.no_grad():
                    action_batch = self.model.predict(
                        obs_tensor_dict, deterministic=False
                    )[0]
                    action_batch = torch.tensor(
                        action_batch, dtype=torch.float32, device=self.model.device
                    )
                    obs_tensor_dict = {
                        key: value.to(self.model.device)
                        for key, value in obs_tensor_dict.items()
                    }
                    q1_batch, q2_batch = self.model.critic(
                        obs_tensor_dict, action_batch
                    )

                    if self.obj_style == "max":
                        mean_vals = (-(q1_batch + q2_batch) / 2).squeeze()
                        sorted_values, sorted_indices = torch.sort(mean_vals)
                        max_index = sorted_indices[-1]
                        sampled_goal_idx = max_index.cpu().item()
                        probs = torch.zeros_like(mean_vals)
                        probs[sampled_goal_idx] = 1

                    elif self.obj_style == "softmax":
                        mean_vals = (-(q1_batch + q2_batch) / 2).squeeze()
                        T = self.temp
                        probs = F.softmax(mean_vals / T)
                        dist = torch.distributions.Categorical(probs)
                        sampled_goal_idx = dist.sample().cpu().item()

                    else:
                        sampled_goal_idx = 0
                        probs = torch.zeros(num_goals)
                        probs[sampled_goal_idx] = 1

                print("Sampling probs: ", probs.detach().cpu().numpy().round(4))
                self.training_env.env_method("set_goal_idx", sampled_goal_idx)
        return True

    def _on_training_start(self) -> None:
        self.tt_sim = 0
        self.full_sim = 0
        self.total_sim = 0


class custom_save_callback(BaseCallback):
    def __init__(self, log_interval, log_path, algo, save=False, verbose=0):
        super(custom_save_callback, self).__init__(verbose)
        self.log_interval = log_interval
        self.log_path = log_path
        self.algo = algo
        self.save = save

    def _on_step(self) -> bool:
        if self.save:
            log_path = os.path.join(self.log_path, "model_" + str(self.num_timesteps))
            if self.num_timesteps % self.log_interval == 0:
                self.model.save(log_path)
                if self.algo == "HER":
                    self.model.save_replay_buffer(log_path)
        return True

    def _on_training_end(self) -> None:
        if self.save:
            log_path = os.path.join(self.log_path, "final_model")
            self.model.save(log_path)
            if self.algo == "HER":
                self.model.save_replay_buffer(log_path)


class custom_eval_callback(BaseCallback):
    def __init__(
        self,
        algo,
        eval_env,
        n_eval_episodes,
        eval_freq,
        traj_len,
        log_path,
        initial_log_timesteps: int = 0,
        verbose: int = 1,
    ):
        super(custom_eval_callback, self).__init__(verbose)
        self.initial_log_timesteps = initial_log_timesteps
        self.last_logged_timesteps = initial_log_timesteps
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.traj_len = traj_len
        self.log_path = log_path
        self.algo = algo
        self.best_success_rate = -np.inf
        self.specs_id = self.eval_env.specs_id
        self.specs_range = self.eval_env.specs_range

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.last_logged_timesteps + self.eval_freq:
            self.last_logged_timesteps = self.num_timesteps
            episode_rewards = np.zeros((self.n_eval_episodes))
            episode_lens = np.zeros((self.n_eval_episodes))
            episode_corner_norm_stds = []
            num_success = 0
            self.eval_env.reset_idx()
            for j in range(self.n_eval_episodes):
                done = False
                reward_total = 0
                corner_dev_total = 0.0
                max_dev_total = 0.0
                success_rewards = []
                eval_obs, _info = self.eval_env.reset()
                print("eval task: ", j + 1)
                for i in range(self.traj_len):
                    eval_action, _states = self.model.predict(
                        eval_obs, deterministic=True
                    )
                    eval_obs, reward, done, truncated, info = self.eval_env.step(
                        eval_action
                    )
                    reward_total += reward
                    cur_specs = info["cur_specs"]

                    if done:
                        success_rewards.append(reward)
                        num_success += 1
                        episode_rewards[j] = reward_total
                        episode_lens[j] = i + 1
                        episode_corner_norm_stds.append(info["ep_corner_norm_std"])
                        print("target specs: ", _info["target_specs"])
                        print("cur specs (succeed): ", cur_specs)
                        print("params: ", info["params"])
                        print("success: ", num_success, "/", (j + 1))
                        print("--------------------------------")
                        break
                    if i == self.traj_len - 1:
                        episode_rewards[j] = reward_total
                        episode_lens[j] = i + 1
                        print("target specs: ", _info["target_specs"])
                        print("cur specs (failed): ", cur_specs)
                        print("params: ", info["params"])
                        print("success: ", num_success, "/", (j + 1))
                        print("--------------------------------")
                        break

            mean_reward = np.mean(episode_rewards)
            mean_len = np.mean(episode_lens)
            if len(episode_corner_norm_stds) == 0:
                mean_corner_norm_std = 1.0
            else:
                mean_corner_norm_std = np.mean(np.array(episode_corner_norm_stds))
            success_rate = num_success / self.n_eval_episodes
            wandb.log(
                {
                    "eval_reward": mean_reward,
                    "eval_len": mean_len,
                    "eval_corner_norm_std": mean_corner_norm_std,
                    "success_rate": success_rate,
                }
            )
        return True

    def _on_training_start(self) -> None:
        self.last_logged_timesteps = self.initial_log_timesteps


@click.command()
# Main options.
@click.option(
    "--algo",
    help="Algorithm to use (PPO, HER, DDPG, SAC)",
    type=click.Choice(["HER", "PPO", "DDPG", "SAC"]),
    default="HER",
    required=True,
)
@click.option(
    "--name",
    help="Name of the run",
    type=str,
    default="default_run_name",
    required=True,
)
@click.option(
    "--seed", help="Random seed (optional)", type=int, default=50, required=True
)
@click.option(
    "--entity",
    help="Wandb entity name",
    type=str,
    default="your_entity_name",
    required=False,
)
@click.option(
    "--env_name",
    help="Environment to use (TwoStageAmp, Gym_FCAmp, comparator)",
    type=str,
    default="COMP",
    required=True,
)
@click.option(
    "--project_name",
    help="Project name for wandb",
    type=str,
    default="comp",
    required=True,
)
@click.option(
    "--yaml",
    "CIR_YAML",
    help="Path to the yaml file",
    type=str,
    default="./eval_engines/ngspice/ngspice_inputs/yaml_files/comparator_cont_gf180_full_2.yaml",
    required=True,
)
@click.option(
    "--spec_path",
    help="Path to the target specs file for training",
    type=str,
    default="ngspice_specs_gen_comp_gf180_350_2",
    required=True,
)
@click.option(
    "--eval_spec_path",
    help="Path to the target specs file for evaluation",
    type=str,
    default="ngspice_specs_gen_comp_gf180_test_20_2",
    required=True,
)
@click.option(
    "--eval_yaml",
    "eval_CIR_YAML",
    help="Path to the yaml file for evaluation",
    type=str,
    default="./eval_engines/ngspice/ngspice_inputs/yaml_files/comparator_cont_gf180_full_3.yaml",
    required=True,
)

# hyperparameters
@click.option(
    "--learning_starts",
    help="Number of steps before starting training",
    type=int,
    default=1000,
    required=True,
)
@click.option(
    "--lr", 
    help="Initial learning rate", 
    type=float, default=3e-3, 
    required=True
)
@click.option(
    "--n_goal",
    "n_sample_goal",
    help="Number of sampled goals",
    type=int,
    default=1,
    required=True,
)
@click.option(
    "--goal",
    "goal_selection_strategy",
    help="Goal selection strategy",
    type=click.Choice(["final", "episode", "future"]),
    default="future",
    required=True,
)
@click.option(
    "--gamma", 
    help="Discount factor",
    type=float, 
    default=0.9, 
    required=True
)
@click.option(
    "--tau",
    help="Target smoothing coefficient",
    type=float,
    default=0.005,
    required=True,
)
@click.option("--batch_size", help="Batch size", type=int, default=256, required=False)
@click.option(
    "--total_timesteps",
    help="Total timesteps for training",
    type=int,
    default=12000,
    required=True,
)
@click.option(
    "--log_interval",
    help="Logging(save model) interval",
    type=int,
    default=6000,
    required=True,
)
@click.option(
    "--generalize",
    help="Generalize the environment",
    type=bool,
    default=True,
    required=False,
)
@click.option(
    "--episode_len", help="Episode length", type=int, default=30, required=False
)
@click.option(
    "--goal_in_obs",
    help="Goal in observation",
    type=bool,
    default=False,
    required=False,
)
@click.option(
    "--done_in_goal", help="Done in goal", type=bool, default=True, required=False
)
@click.option(
    "--corner_selection",
    help="Corner selection strategy",
    type=str,
    default="worst",
    required=False,
)
@click.option(
    "--state_option",
    help="State option",
    type=str,
    default="worst_spec",
    required=False,
)
@click.option(
    "--SoF",
    help="Skip on Fail",
    type=bool,
    default=True,
    required=False,
)
@click.option(
    "--lookup_style",
    help="Lookup style",
    type=click.Choice(["normd", "tanh"]),
    default="tanh",
    required=False,
)
@click.option(
    "--min_threshold",
    help="Minimum threshold",
    type=float,
    default=-6.0,
    required=False,
)
@click.option(
    "--tt_threshold", help="TT threshold", type=float, default=-1.0, required=False
)
@click.option(
    "--tt_threshold_2",
    help="Conservative TT threshold",
    type=float,
    default=-3.0,
    required=False,
)
@click.option("--concat_all_specs", help="Concat all specs", type=bool, default=False, required=False)
@click.option(
    "--conservative",
    help="conservative virtual reward",
    type=bool,
    default=True,
    required=False,
)
@click.option(
    "--alpha",
    help="regularization coefficient to minimize corner variation",
    type=float,
    default=0.1,
    required=False,
)
@click.option(
    "--online_goal",
    help="Sample goal from target range every episode",
    type=bool,
    default=True,
    required=False,
)
@click.option(
    "--eval_freq", help="Evaluation frequency", type=int, default=6000, required=False
)
@click.option(
    "--num_eval",
    help="Number of evaluation episodes",
    type=int,
    default=20,
    required=False,
)
@click.option(
    "--pareto_freq",
    help="Pareto goal sampling frequency",
    type=int,
    default=1,
    required=False,
)
@click.option(
    "--obj_style", help="inverse probability", type=str, default="softmax", required=False
)
@click.option(
    "--temp", help="sampling temperature", type=float, default=10.0, required=False
)
@click.option("--n_warmup", help="warmup size", type=int, default=2, required=False)

def main(
    algo,
    name,
    seed,
    entity,
    env_name,
    project_name,
    CIR_YAML,
    spec_path,
    eval_spec_path,
    eval_CIR_YAML,
    num_eval,
    eval_freq,
    learning_starts,
    lr,
    n_sample_goal,
    goal_selection_strategy,
    gamma,
    tau,
    total_timesteps,
    log_interval,
    obj_style,
    temp,
    batch_size=256,
    **kwargs,
):
    # training env setting
    env_config = {key: value for key, value in kwargs.items() if value is not None}
    env_config["run_valid"] = False
    env_config["CIR_YAML"] = CIR_YAML
    env_config["spec_path"] = spec_path
    if algo == "HER":
        env_config["relabel_goal"] = True
    else:
        env_config["relabel_goal"] = False
    # eval env setting
    eval_env_config = env_config.copy()
    eval_env_config["generalize"] = True
    eval_env_config["run_valid"] = True
    eval_env_config["spec_path"] = eval_spec_path
    eval_env_config["CIR_YAML"] = eval_CIR_YAML
    eval_env_config["online_goal"] = False

    if algo == "HER":
        sim_env = ngspice_env_goal
    else:
        sim_env = ngspice_env_cont

    envs = sim_env(env_config=env_config)
    eval_envs = sim_env(env_config=eval_env_config)

    envs.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)

    envs.tt_sim = 0
    envs.full_sim = 0

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256, 256, 256], qf=[256, 256, 128])
    )  # Actor and Critic networks

    wandb.tensorboard.patch(root_logdir="./wandb_results/")
    run = wandb.init(
        project=project_name,
        entity=entity,
        name=name,
        config={
            "env_name": env_name,
            "algo": algo,
            "policy_kwargs": policy_kwargs,
            "env_config": env_config,
            "initial learning_rate": lr,
            "n_sample_goal": n_sample_goal,
            "goal_selection_strategy": goal_selection_strategy,
            "gamma": gamma,
            "tau": tau,
            "power_opt": False,
        },
        dir="./wandb_results",
        reinit=True,
        sync_tensorboard=True,
    )

    if algo == "SAC":
        model = SAC(
            "MlpPolicy",
            envs,
            policy_kwargs=policy_kwargs,
            learning_rate=lr,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            tensorboard_log="./wandb_results/sac/",
            verbose=1,
            seed=seed,
        )
    elif algo == "HER":
        print("HER, goal_selection_strategy: ", goal_selection_strategy)
        print("exploration steps before training: ", learning_starts)
        model = SAC(
            "MultiInputPolicy",
            envs,
            policy_kwargs=policy_kwargs,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=n_sample_goal,
                goal_selection_strategy=goal_selection_strategy,
                copy_info_dict=True,
            ),
            learning_rate=lr,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            tensorboard_log="./wandb_results/her/",
            verbose=1,
            seed=seed,
        )
    elif algo == "DDPG":
        model = DDPG(
            "MlpPolicy",
            envs,
            policy_kwargs=policy_kwargs,
            learning_rate=lr,
            batch_size=batch_size,
            learning_starts=learning_starts,
            gamma=gamma,
            tau=tau,
            tensorboard_log="./wandb_results/ddpg/",
            verbose=1,
            seed=seed,
        )
    elif algo == "PPO":
        model = PPO(
            "MlpPolicy",
            envs,
            learning_rate=lr,
            n_steps=360,
            batch_size=120,
            gamma=0.9,
            gae_lambda=0.85,
            tensorboard_log="./wandb_results/ppo/",
            verbose=1,
            seed=seed,
        )

    tb_log_base = (
        f"{algo}_{env_name}_early_done_{name}_gf180_pipelined_goal_new2_full_deepernn"
        f"_lr{lr}_gamma{gamma}_nsg{n_sample_goal}_{goal_selection_strategy}_tau{tau}_seed{seed}"
    )
    #########load model #######
    # model = SAC.load(tb_log_base, env=envs)
    # model.load_replay_buffer(tb_log_base)
    #########training#######
    tb_log_name = tb_log_base + "_" + str(total_timesteps)
    model_dir = os.path.join(os.getcwd(), "models", tb_log_base)

    os.makedirs(model_dir, exist_ok=True)
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=algo + "_" + env_name + "_" + str(total_timesteps),
        reset_num_timesteps=False,
        callback=CallbackList(
            [
                WandbCallback(verbose=2),
                custom_eval_callback(
                    algo=algo,
                    eval_env=eval_envs,
                    n_eval_episodes=num_eval,
                    eval_freq=eval_freq,
                    traj_len=env_config["episode_len"],
                    log_path=model_dir,
                    initial_log_timesteps=0,
                ),
                train_callback(
                    num_corners=envs.num_corners,
                    env=envs,
                    obj_style=obj_style,
                    temp=temp,
                    algo=algo,
                    pareto_freq=env_config["pareto_freq"],
                ),
                custom_save_callback(log_interval, model_dir, algo),
            ]
        ),
    )

    run.finish()
    del model
    print("Finised training model: ", tb_log_name)


# ------------------------------------------------
if __name__ == "__main__":
    main()
# ------------------------------------------------
