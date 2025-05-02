"""
A new ckt environment based on a new structure of MDP
"""
import gymnasium
from gymnasium import spaces

import numpy as np
import random
import torch
import math


from multiprocessing.dummy import Pool as ThreadPool
from collections import OrderedDict
import yaml
import yaml.constructor
import statistics
import os
import IPython
import itertools
import sys
from eval_engines.util.core import *
import pickle
import os
from PPAAS.envs.ngspice_env import ngspice_env_cont
from eval_engines.ngspice.CircuitClass import *
import pdb

verbose = False

class ngspice_env_goal(ngspice_env_cont):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_config):
        super().__init__(env_config)
        self.goal_in_obs = env_config.get("goal_in_obs", True)
        self.done_in_goal = env_config.get("done_in_goal", True)
        self.relabel_goal = env_config.get("relabel_goal", True)
        self.conservative = env_config.get("conservative", True)
        
        if self.goal_in_obs:
            if self.done_in_goal:
                obs = spaces.Box(
                    low=np.array([-np.inf] * 2 * len(self.specs_id) + 2*[0.0] + len(self.params_id) * [-np.inf]),
                    high=np.array([np.inf] * 2 * len(self.specs_id) + 2*[1.0] + len(self.params_id) * [np.inf]),
                    dtype=np.float64
                )
            else:
                obs = spaces.Box(
                    low=np.array([-np.inf] * 2 * len(self.specs_id) + len(self.params_id) * [-np.inf]),
                    high=np.array([np.inf] * 2 * len(self.specs_id) + len(self.params_id) * [np.inf]),
                    dtype=np.float64
                )
        else:
            obs = spaces.Box(
                low=np.array([-np.inf]*len(self.params_id)),
                high=np.array([np.inf]*len(self.params_id)),
                dtype=np.float64
            )
        if self.done_in_goal:
            achieved_goal = spaces.Box(
                low=np.array([-np.inf]*len(self.specs_id)+2*[0.0]),
                high=np.array([np.inf]*len(self.specs_id)+2*[1.0]),
                dtype=np.float64
            )
            desired_goal = spaces.Box(
                low=np.array([-np.inf]*len(self.specs_id)+2*[0.0]),
                high=np.array([np.inf]*len(self.specs_id)+2*[1.0]),
                dtype=np.float64
            )
        else:
            achieved_goal = spaces.Box(
                low=np.array([-np.inf]*len(self.specs_id)),
                high=np.array([np.inf]*len(self.specs_id)),
                dtype=np.float64
            )
            desired_goal = spaces.Box(
                low=np.array([-np.inf]*len(self.specs_id)),
                high=np.array([np.inf]*len(self.specs_id)),
                dtype=np.float64
            )
        self.observation_space = spaces.Dict(
            {
                "observation": obs,
                "achieved_goal": achieved_goal,
                "desired_goal": desired_goal
            }
        )
        self.get_from_info = True
        print("environment initialize")
        print("observation_space: ", self.observation_space)
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)
        if verbose:
            print("-"*20)
            print(info)
            print("-"*20)
        self.gc_obs_init = self.translate_obs(obs, info)
        return self.gc_obs_init, info
    
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        obs = self.translate_obs(obs, info)
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        if verbose:
            print("-"*20)
            print(info)
            print("_"*20)
        return obs, reward, done, truncated, info
    
    def get_obs(self):
        return self.gc_obs_init
    
    
    

    def compute_reward(self, achieved_goal, desired_goal, infos):
        """
        Compute the reward tensor for a batch of goal‐conditioned transitions,
        with optional HER relabeling, binary rewards, and UVFA default.
        Returns a scalar if inputs were unbatched.
        """
        # real transition
        if isinstance(infos, dict):
            return infos["reward"]
        
        # virtual transition (batch)
        device = getattr(self, "device", None) or \
                 torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ag = torch.as_tensor(achieved_goal, device=device, dtype=torch.float32)
        dg = torch.as_tensor(desired_goal,  device=device, dtype=torch.float32)

        if not self.relabel_goal:
            # Pull environment reward directly
            base = [info["reward"] for info in infos]
            rewards = torch.as_tensor(base, device=device, dtype=torch.float32)
            return rewards
        
        cur_norm = ag[:, :-2]       # (batch, m)
        goal_norm = dg[:, :-2]      # (batch, m)
        done      = ag[:, -2]       # (batch,)
        tt_done   = ag[:, -1]       # (batch,)
        
        copy_info_dict = False #TODO: remove this
        if copy_info_dict: 
            cur_specs = torch.stack([
                torch.as_tensor(info["cur_specs"], device=device, dtype=torch.float32)
                for info in infos
            ], dim=0)
            goal_specs = torch.stack([
                torch.as_tensor(info["target_specs"], device=device, dtype=torch.float32)
                for info in infos
            ], dim=0)
        else: 
            cur_specs   = self.unlookup(cur_norm, self.global_g)
            goal_specs  = self.unlookup(goal_norm, self.global_g)

        all_specs = torch.stack([
            torch.as_tensor(info["all_specs"], device=device, dtype=torch.float32)
            for info in infos
        ], dim=0) 

        # Compute base reward via your torchified reward_batch()
        base_r = self.reward_batch(cur_specs, goal_specs)  # (batch,)

        # Corner std‐deviation penalty
        corner_std = torch.sqrt(
            torch.mean((all_specs[:,1:]/all_specs[:,[0]] - 1.0).square(), dim=(1,2))
        ).clamp(0.0, 1.0)  # (batch,)


        # 12) Build boolean masks
        done_mask    = done > 0.5
        tt_mask      = tt_done > 0.5


        if self.conservative:
            reward = torch.where(
                done_mask,
                torch.full_like(base_r, 30.0),
                torch.where(
                    tt_mask,
                    (0.0 - self.tt_threshold_2) * (base_r / len(self.specs_id))
                    - self.alpha * corner_std,
                    self.tt_threshold_2
                    + (self.tt_threshold_2 - self.min_threshold)
                      * (base_r / len(self.specs_id))
                    - self.alpha
                )
            )
        else:
            reward = torch.where(
                done_mask,
                torch.full_like(base_r, 30.0),
                torch.where(
                    tt_mask,
                    (0.0 - self.tt_threshold) * (base_r / len(self.specs_id))
                    - self.alpha * corner_std,
                    self.tt_threshold
                    + (self.tt_threshold - self.min_threshold)
                      * (base_r / len(self.specs_id))
                    - self.alpha
                )
            )

        return reward.cpu().numpy()


    def unlookup(self, norm_spec: torch.Tensor, goal_spec) -> torch.Tensor:
        # Ensure goal_spec is a torch Tensor on the same device/dtype
        if not isinstance(goal_spec, torch.Tensor):
            goal_spec = torch.as_tensor(goal_spec,
                                        device=norm_spec.device,
                                        dtype=norm_spec.dtype)

        if self.lookup_style == "normd":
            return -(norm_spec + 1.0) * goal_spec / (norm_spec - 1.0)

        elif self.lookup_style == "tanh":
            scale = getattr(self, "scale_factor", 10.0)
            f = math.tanh(1.0 / scale)
            x = torch.clamp(norm_spec * f, -0.999, 0.999)
            atanh_x = 0.5 * (torch.log1p(x) - torch.log1p(-x))
            return goal_spec * (1.0 + scale * atanh_x)

        else:
            raise ValueError(f"Unknown lookup_style {self.lookup_style!r}")

    def reward_batch(self, spec_batch: torch.Tensor, goal_spec_batch: torch.Tensor) -> torch.Tensor:
        """
        GPU-accelerated reward computation with inline lookup (normalized_difference or tanh).
        spec_batch, goal_spec_batch: tensors of shape (batch_size, m) on the correct device.
        Returns: tensor of shape (batch_size,)
        """
        device = spec_batch.device
        epsilon = 1e-9
    
        spec = spec_batch
        goal = goal_spec_batch
        # avoid division by zero
        goal_safe = torch.where(goal == 0.0,
                                torch.full_like(goal, epsilon),
                                goal)
    
        if self.lookup_style == "normd":
            delta = spec - goal_safe
            abs_delta = delta.abs() + epsilon
            denom = goal_safe + spec.abs() + epsilon
            rel_specs = torch.sign(delta) * torch.exp(torch.log(abs_delta) - torch.log(denom))
    
        elif self.lookup_style == "tanh":
            scale = getattr(self, "scale_factor", 10.0)
            raw = (spec - goal_safe) / (scale * goal_safe)
            rel_specs = torch.tanh(raw) / math.tanh(1.0 / scale)
    
        else:
            raise ValueError(f"Unknown lookup_style {self.lookup_style!r}")
    
        # --- Adjustment factors for "max" specs ---
        m = rel_specs.size(1)
        adj = torch.ones((m,), device=device)
        for i, sid in enumerate(self.specs_id):
            if sid.endswith("max"):
                adj[i] = -1.0
        adjusted = rel_specs * adj.unsqueeze(0)
    
        # --- Sign-mismatch penalty ---
        sign_mismatch = (spec < 0.0) != (goal < 0.0)
        adjusted = torch.where(sign_mismatch,
                               torch.full_like(adjusted, -1.0),
                               adjusted)
    
        negative_mask = adjusted < 0.0
        summed = torch.sum(torch.where(negative_mask,
                                       adjusted,
                                       torch.zeros_like(adjusted)),
                           dim=1)
    
        rewards = torch.where(summed < -0.02,
                              summed,
                              torch.full_like(summed, 10.0))
    
        return rewards
        
    def translate_obs(self, obs, info):
        params = obs[-len(self.params_id):]
        achieved_goal = obs[:len(self.specs_id)]
        desired_goal = obs[len(self.specs_id):-len(self.params_id)]
        if self.done_in_goal:
            if info["done"] is not None and info["tt_done"] is not None:
                done = [info["done"], info["tt_done"]]
                done_score = np.where(done, 1.0, 0.0)
                obs = np.concatenate((achieved_goal, desired_goal, done_score, params))
                achieved_goal = np.concatenate((achieved_goal, done_score))
                desired_goal = np.concatenate((desired_goal, np.array([1.0, 1.0])))
            else:
                raise ValueError("done and tt_done must be in info")
        if self.goal_in_obs:
            obs_with_goal = {"observation": obs, "achieved_goal": achieved_goal, "desired_goal": desired_goal}
        else:
            obs_with_goal = {"observation": params, "achieved_goal": achieved_goal, "desired_goal": desired_goal}
        return obs_with_goal
                        