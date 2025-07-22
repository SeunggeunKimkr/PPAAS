import gymnasium
from gymnasium import spaces
import numpy as np
import random
from collections import OrderedDict
import yaml
import yaml.constructor
import os
from eval_engines.util.core import *
import pickle
import os
import pdb
import copy
from abc import ABC, abstractmethod
from eval_engines.ngspice.CircuitClass import *

#way of ordering the way a yaml file is read
class OrderedDictYAMLLoader(yaml.Loader):
    """
    A YAML loader that loads mappings into ordered dictionaries.
    """

    def __init__(self, *args, **kwargs):
        yaml.Loader.__init__(self, *args, **kwargs)

        self.add_constructor(u'tag:yaml.org,2002:map', type(self).construct_yaml_map)
        self.add_constructor(u'tag:yaml.org,2002:omap', type(self).construct_yaml_map)

    def construct_yaml_map(self, node):
        data = OrderedDict()
        yield data
        value = self.construct_mapping(node)
        data.update(value)

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
        else:
            raise yaml.constructor.ConstructorError(None, None,
                                                    'expected a mapping node, but found %s' % node.id, node.start_mark)

        mapping = OrderedDict()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping

class ngspice_env(gymnasium.Env, ABC):
    metadata = {'render.modes': ['human']}

    ACT_LOW = -1
    ACT_HIGH = 1

    #obtains yaml file
    path = os.getcwd()
    CIR_YAML = path+'/eval_engines/ngspice/ngspice_inputs/yaml_files/two_stage_opamp.yaml'

    def __init__(self, env_config):
        self.generalize = env_config.get("generalize",True)
        self.episode_len = env_config.get("episode_len", 30)
        self.valid = env_config.get("run_valid", False)
        self.CIR_YAML = env_config.get("CIR_YAML", ngspice_env.CIR_YAML)
        self.spec_path = env_config.get("spec_path", "ngspice_specs_gen_two_stage_opamp")
        self.SoF = env_config.get("SoF", True)
        self.lookup_style = env_config.get("lookup_style", "normd")
        self.min_threshold = env_config.get("min_threshold", -6.0)
        self.tt_threshold = env_config.get("tt_threshold", -1.0)
        self.tt_threshold_2 = env_config.get("tt_threshold_2", -3.0)
        self.concat_all_specs = env_config.get("concat_all_specs", False)
        self.verbose = env_config.get("verbose", False)
        self.alpha = env_config.get("alpha", 0.1)
        self.online_goal = env_config.get("online_goal", False)
        self.pareto_freq = env_config.get("pareto_freq", 0)
        self.n_warmup = env_config.get("n_warmup", 4)
        self.env_steps = 0
        with open(self.CIR_YAML, 'r') as f:
            yaml_data = yaml.load(f, OrderedDictYAMLLoader)
        self.yaml_data = yaml_data

        # single goal
        if self.generalize == False:
            specs = yaml_data['target_specs']
        else:  # multi goals
            if (self.online_goal == False):
                load_specs_path = ngspice_env.path + "/PPAAS/gen_specs/" + self.spec_path
                with open(load_specs_path, 'rb') as f:
                    specs = pickle.load(f)
            else:
                specs = yaml_data['target_specs']
        self.specs = OrderedDict(sorted(specs.items(), key=lambda k: k[0]))
        
        self.specs_ideal = []
        self.specs_id = list(self.specs.keys())
        self.num_os = len(list(self.specs.values())[0])
        self.cur_steps = 0
        self.full_sim = 0
        self.tt_sim = 0
        self.horizon = self.episode_len
        self.num_corners = len(yaml_data['dsn_netlist'])
        
        # param array
        params = yaml_data['params']
        self.params_id = list(params.keys())
        self.params_val = list(params.values())
        
        # goal history buffer
        self.pareto_goal_history = []
        self.rejection = 0
        self.episode_steps = 0
        

        specs_range_dict = OrderedDict(sorted(self.yaml_data["target_specs"].items(), key=lambda k:k[0]))
        self.specs_range = list(specs_range_dict.values())
        self.episode_corner_norm_std = 1.0

        #initialize sim environment
        tt_design_netlist = [yaml_data['dsn_netlist'][0]]
        corner_design_netlists = yaml_data['dsn_netlist'][1:]
        self.corner_sim_env = CircuitClass(yaml_path=self.CIR_YAML, path=ngspice_env.path, design_netlists=corner_design_netlists)
        self.tt_sim_env = CircuitClass(yaml_path=self.CIR_YAML, path=ngspice_env.path, design_netlists=tt_design_netlist)
        self.full_sim_env = CircuitClass(yaml_path=self.CIR_YAML, path=ngspice_env.path)
        
        if env_config.get("action_type")=="discrete":
            self.action_meaning = [-1, 0, 2]
            self.action_space = spaces.MultiDiscrete(
                [len(self.action_meaning)] * len(self.params_id)
            )
        elif env_config.get("action_type")=="continuous":
            self.action_space = spaces.Box(
                low=np.array([ngspice_env.ACT_LOW] * len(self.params_id)),
                high=np.array([ngspice_env.ACT_HIGH] * len(self.params_id)),
            )
            
        if self.concat_all_specs:
            self.observation_space = spaces.Box(
                low=np.array(
                    [-np.inf] * (self.num_corners + 1) * len(self.specs_id) + len(self.params_id)*[-np.inf]
                ),
                high=np.array(
                    [np.inf] * (self.num_corners + 1) * len(self.specs_id) + len(self.params_id)*[np.inf]
                ),
                dtype=np.float64
            )
            self.cur_specs = np.zeros(len(self.specs_id)*self.num_corners, dtype=np.float64)
        else:
            self.observation_space = spaces.Box(
                low=np.array(
                    [-np.inf] * 2 * len(self.specs_id) + len(self.params_id)*[-np.inf]
                ),
                high=np.array(
                    [np.inf] * 2 * len(self.specs_id) + len(self.params_id)*[np.inf]
                ),
                dtype=np.float64
            )     
            self.cur_specs = np.zeros(len(self.specs_id), dtype=np.float64)
        
        if env_config.get("action_type") == "discrete":
            self.cur_params = np.zeros(len(self.params_id), dtype=np.int32)
        elif env_config.get("action_type") == "continuous":
            self.cur_params = np.zeros(len(self.params_id), dtype=np.float64)

        #Get the g* (overall design spec) you want to reach
        self.g_star = np.array(yaml_data['normalize'])
        self.global_g = np.array(yaml_data['normalize'])
        
        #objective number (used for exploitation)
        self.obj_idx = 0
    
    def reset(self, seed=None, options=None):
        #if multi-goal is selected, every time reset occurs, it will select a different design spec as objective
        if self.generalize == True:
            if self.online_goal:
                # Only perform PGDS if history has sufficient data
                if len(self.pareto_goal_history) >= self.n_warmup and self.pareto_freq > 0:
                    if self.episode_steps % self.pareto_freq == 0:
                        self.specs_ideal=self.specs_ideal_candidates[self.goal_idx]
                    else:
                        self.specs_ideal = self.sample_goal_uniform(list(self.specs.values()), num_goals_to_sample=1)[0]
                    self.episode_steps += 1
                else:
                    # Initially random sampling
                    self.specs_ideal = self.sample_goal_uniform(list(self.specs.values()), num_goals_to_sample=1)[0]
            else:
                if self.valid == True:
                    if self.obj_idx > self.num_os-1:
                        self.obj_idx = 0
                    idx = self.obj_idx
                    self.obj_idx += 1
                else:
                    idx = random.randint(0,self.num_os-1)
                self.specs_ideal = []
                for spec in list(self.specs.values()):
                    self.specs_ideal.append(spec[idx])
                self.specs_ideal = np.array(self.specs_ideal)
        else:
            self.specs_ideal = self.g_star 
            
        self.specs_ideal_norm = self.lookup(self.specs_ideal, self.global_g)

        self.cur_steps = 0
        self.cur_params = self.init_params()
        self.full_sim = 0
        self.tt_sim = 0
        tt_done = False
        min_threshold = self.min_threshold
        tt_threshold = self.tt_threshold
        if self.SoF:
            # 1st stage: TT corner simulation
            self.cur_specs = self.update(self.cur_params, self.tt_sim_env)[0]
            cur_spec_norms = []
            cur_spec_norm = self.lookup(self.cur_specs, self.global_g)
            all_specs = np.array([self.cur_specs]*self.num_corners)
            spec_diff = self.aggregate(self.cur_specs, self.specs_ideal)
            if spec_diff < 0:
                reward = tt_threshold + (tt_threshold - min_threshold) * (spec_diff / float(len(self.specs_id)))
            # 2nd stage: full corner simulation
            else: 
                tt_done = True
                full_cur_specs = self.update(self.cur_params, self.corner_sim_env)
                for cur_spec in full_cur_specs:
                    cur_spec_norms.append(self.lookup(cur_spec, self.global_g))
                cur_spec_norms = np.array([cur_spec_norm] + cur_spec_norms)
                reverse_indices = []
                for i in range(len(self.specs_id)):
                    if(self.specs_id[i][-3:] == 'max'):
                        cur_spec_norms[:,i] = cur_spec_norms[:,i]*-1.0
                        reverse_indices.append(i)
                cur_spec_norm = np.min(cur_spec_norms, axis=0)
                worst_idx = np.argmin(cur_spec_norms, axis=0)
                reverse_indices = np.array(reverse_indices)
                cur_spec_norm[reverse_indices] = -cur_spec_norm[reverse_indices]
                self.cur_specs = np.array([self.cur_specs] + full_cur_specs)
                all_specs = copy.deepcopy(self.cur_specs)
                self.cur_specs = self.cur_specs[worst_idx, np.arange(len(worst_idx))]
                spec_diff = self.aggregate(self.cur_specs, self.specs_ideal)
                if spec_diff < 0:
                    reward = -tt_threshold * spec_diff / float(len(self.specs_id))
                
                #3rd stage: satified all desired specs
                else:
                    reward = 30.0
        
        else:
            tt_done = False
            self.cur_specs = self.update(self.cur_params, self.full_sim_env)
            cur_spec_norms = []
            for cur_spec in self.cur_specs:
                cur_spec_norms.append(self.lookup(cur_spec, self.global_g))
            cur_spec_norms = np.array(cur_spec_norms)
            reverse_indices = []
            for i in range(len(self.specs_id)):
                if(self.specs_id[i][-3:] == 'max'):
                    cur_spec_norms[:,i] = cur_spec_norms[:,i]*-1.0
                    reverse_indices.append(i)

            cur_spec_norm = np.min(cur_spec_norms, axis=0)
            worst_idx = np.argmin(cur_spec_norms, axis=0)
            reverse_indices = np.array(reverse_indices)
            cur_spec_norm[reverse_indices] = -cur_spec_norm[reverse_indices]
            self.cur_specs = np.array(self.cur_specs)
            self.cur_specs = self.cur_specs[worst_idx, np.arange(len(worst_idx))]
            spec_diff = self.aggregate(self.cur_specs, self.specs_ideal)
            if self.concat_all_specs:
                cur_spec_norm = cur_spec_norms.flatten()
            if spec_diff < 0:
                reward = -min_threshold * spec_diff / float(len(self.specs_id))
            else:
                reward = 30.0
                
        if reward >= 0:
            done = True
        else:
            done = False
        truncated = False
        
        info = {
            "reward": reward,
            "params": self.cur_params,
            "target_specs": self.specs_ideal,
            "cur_specs": self.cur_specs,
            "all_specs": all_specs,
            "done": done,
            "truncated": truncated,
            "tt_done": tt_done,
            "pareto_buffer_size": len(self.pareto_goal_history),
        }
        self.ob = np.concatenate([cur_spec_norm, self.specs_ideal_norm, self.cur_params])
        return self.ob, info

    def step(self, action):
        action = list(np.reshape(np.array(action),(np.array(action).shape[0],)))
        self.cur_params = self.update_params(action)
        tt_done = False
        worst_idx = [0] * len(self.specs_id)
        min_threshold = self.min_threshold
        tt_threshold = self.tt_threshold
        if self.SoF:
            # 1st stage: TT corner simulation
            self.cur_specs = self.update(self.cur_params, self.tt_sim_env)[0]
            cur_spec_norms = []
            cur_spec_norm = self.lookup(self.cur_specs, self.global_g)
            all_specs = np.array([self.cur_specs]*self.num_corners)
            spec_diff = self.aggregate(self.cur_specs, self.specs_ideal)
            if spec_diff < 0:
                tt_done = False
                self.tt_sim += 1
                reward = tt_threshold + (tt_threshold - min_threshold) * (spec_diff / float(len(self.specs_id)))
                corner_norm_std = 1.0
            else:
                # 2nd stage: full corner simulation
                tt_done = True
                self.full_sim += 1
                full_cur_specs = self.update(self.cur_params, self.corner_sim_env)
                for cur_spec in full_cur_specs:
                    cur_spec_norms.append(self.lookup(cur_spec, self.global_g))
                cur_spec_norms = np.array([cur_spec_norm] + cur_spec_norms)
                reverse_indices = []
                for i in range(len(self.specs_id)):
                    if(self.specs_id[i][-3:] == 'max'):
                        cur_spec_norms[:,i] = cur_spec_norms[:,i]*-1.0
                        reverse_indices.append(i)
                cur_spec_norm = np.min(cur_spec_norms, axis=0)
                worst_idx = np.argmin(cur_spec_norms, axis=0)
                reverse_indices = np.array(reverse_indices)
                cur_spec_norm[reverse_indices] = -cur_spec_norm[reverse_indices]
                cur_spec_norms[:,reverse_indices] = -cur_spec_norms[:,reverse_indices]
                self.cur_specs = np.array([self.cur_specs] + full_cur_specs)
                all_specs = copy.deepcopy(self.cur_specs)
                self.cur_specs = self.cur_specs[worst_idx, np.arange(len(worst_idx))]
                spec_diff = self.aggregate(self.cur_specs, self.specs_ideal)
                corner_norm_std = np.sqrt(np.mean((all_specs[1:]/all_specs[0]-1)**2))
                corner_norm_std = np.clip(corner_norm_std, 0, 1)
                if spec_diff < 0:
                    reward = -tt_threshold * spec_diff / float(len(self.specs_id))
                else:
                    reward = 30.0
        
        else:
            self.full_sim += 1
            self.cur_specs = self.update(self.cur_params, self.full_sim_env)
            cur_spec_norms = []
            for cur_spec in self.cur_specs:
                cur_spec_norms.append(self.lookup(cur_spec, self.global_g))
            cur_spec_norms = np.array(cur_spec_norms)
            reverse_indices = []
            for i in range(len(self.specs_id)):
                if(self.specs_id[i][-3:] == 'max'):
                    cur_spec_norms[:,i] = cur_spec_norms[:,i]*-1.0
                    reverse_indices.append(i)

            cur_spec_norm = np.min(cur_spec_norms, axis=0)
            worst_idx = np.argmin(cur_spec_norms, axis=0)
            reverse_indices = np.array(reverse_indices)
            cur_spec_norm[reverse_indices] = -cur_spec_norm[reverse_indices]
            self.cur_specs = np.array(self.cur_specs)
            all_specs = copy.deepcopy(self.cur_specs)
            self.cur_specs = self.cur_specs[worst_idx, np.arange(len(worst_idx))]
            spec_diff = self.aggregate(self.cur_specs, self.specs_ideal)
            corner_norm_std = np.sqrt(np.mean((all_specs[1:]/all_specs[0]-1)**2))
            corner_norm_std = np.clip(corner_norm_std, 0, 1)
            if self.concat_all_specs:
                cur_spec_norm = cur_spec_norms.flatten()
            if spec_diff < 0:
                reward = -min_threshold * spec_diff / float(len(self.specs_id))
            else:
                reward = 30.0
        
        #add deviation penalty
        reward = reward - self.alpha * corner_norm_std
            
        
        self.env_steps = self.env_steps + 1
        self.cur_steps = self.cur_steps + 1
        
        if reward >= 0:
            done = True
        else:
            done = False
        
        if self.cur_steps >= self.horizon:
            truncated = True
        else:    
            truncated = False


        if done and self.online_goal:
            self.pareto_goal_history = self.update_pareto_goals(self.specs_ideal, self.pareto_goal_history)
        
        if done or truncated:
            self.episode_corner_norm_std = corner_norm_std
            if self.online_goal:
                if self.pareto_freq > 0:
                    self.specs_ideal_candidates = self.sample_goal_pareto(self.pareto_goal_history, num_goals_to_sample=16)
                else:
                    self.specs_ideal_candidates = self.sample_goal_uniform(list(self.specs.values()), num_goals_to_sample=16)
                self.goals_norm = self.lookup(self.specs_ideal_candidates, self.global_g)
        
        self.ob = np.concatenate([cur_spec_norm, self.specs_ideal_norm, self.cur_params])
        info = {
            "reward": reward,
            "params": self.cur_params,
            "target_specs": self.specs_ideal,
            "cur_specs": self.cur_specs,
            "all_specs": all_specs,
            "worst_index": worst_idx,
            "done": done,
            "tt_done": tt_done,
            "truncated": truncated,
            "full_sim": self.full_sim,
            "tt_sim": self.tt_sim,
            "ep_corner_norm_std": self.episode_corner_norm_std,
            "pareto_buffer_size": len(self.pareto_goal_history),
        }
        return self.ob, reward, done, truncated, info

    def lookup(self, spec, goal_spec):
        spec = np.asarray(spec, dtype=np.float32)
        goal_spec = np.asarray(goal_spec, dtype=np.float32)
        epsilon = 1e-9
        goal_spec = np.where(goal_spec == 0, epsilon, goal_spec) 
        if self.lookup_style == "normd":         
            delta = spec - goal_spec
            abs_delta = np.abs(delta) + epsilon  # Avoid log(0)
            denom = goal_spec + np.abs(spec) + epsilon  # Avoid log(0)
            norm_spec = np.sign(delta) * np.exp(np.log(abs_delta) - np.log(denom))

        elif self.lookup_style == "tanh":
            try:
                scale_factor = 10.0
                norm_spec = np.tanh((spec - goal_spec) / (scale_factor* goal_spec))/ np.tanh(1/scale_factor)
            except RuntimeWarning as e:
                pdb.set_trace()
        return norm_spec
    
    def unlookup(self, norm_spec, goal_spec):
        if self.lookup_style == "normd":
            spec = -1*np.multiply((norm_spec+1), goal_spec)/(norm_spec-1)
        elif self.lookup_style == "tanh":
            try:
                scale_factor = 10.0
                x = np.clip(norm_spec*np.tanh(1/scale_factor), -0.999, 0.999)
                spec = goal_spec * (1 + scale_factor * np.arctanh(x))
            except RuntimeWarning as e:
                pdb.set_trace()
        return spec
        
    def aggregate(self, spec, goal_spec):
        rel_specs = self.lookup(spec, goal_spec)
        pos_val = [] 
        reward = 0.0
        for i,rel_spec in enumerate(rel_specs):
            if(self.specs_id[i][-3:] == 'max'):
                rel_spec = rel_spec*-1.0
            if rel_spec < 0:
                reward += rel_spec
                pos_val.append(0)
            else:
                pos_val.append(1)
        return reward if reward <-0.02 else 10.0 
            
    def update(self, cur_params, sim_env):
        params = self.translate_params(cur_params)
        param_val = [OrderedDict(list(zip(self.params_id,params)))]
        states, specs, infos = sim_env.run(param_val[0])
        cur_specs = []
        for spec in specs:
            cur_spec = OrderedDict(sorted(spec.items(), key=lambda k:k[0]))
            cur_specs.append(np.array(list(cur_spec.values())))
        return cur_specs
            
    def set_goal_idx(self, goal_idx):
        self.goal_idx=goal_idx
        return True
    
    def sample_goal_pareto(self, pareto_goals, num_goals_to_sample=1):
        sampled_goals = []
        while len(sampled_goals) < num_goals_to_sample:
            goal = self.sample_goal_uniform(self.specs_range, num_goals_to_sample=1)[0]
            dominated, dominant = self.is_goal_dominated(goal, pareto_goals)
            if not dominated:
                sampled_goals.append(goal)
                self.rejection = 0
            else:
                self.rejection += 1
        return np.array(sampled_goals)

    def sample_goal_uniform(self, specs_range_vals, num_goals_to_sample=1):
            sampled_goals = []
            while len(sampled_goals) < num_goals_to_sample:    
                specs_valid = []
                for spec in specs_range_vals:
                    if isinstance(spec[0], int):
                        specs_valid.append(random.randint(int(spec[0]), int(spec[1])))
                    else:
                        specs_valid.append(random.uniform(float(spec[0]), float(spec[1])))
                sampled_goals.append(specs_valid)
            return np.array(sampled_goals)
    
    def update_pareto_goals(self, new_goal, pareto_goals):
        dominated, _ = self.is_goal_dominated(new_goal, pareto_goals)
        if dominated:
            return pareto_goals
        # Remove any existing goals that the new goal dominates
        pareto_goals = [goal for goal in pareto_goals if not self.is_dominated(goal, new_goal)]
        pareto_goals.append(new_goal)
        return pareto_goals

    def is_goal_dominated(self, goal, goal_set):
        """Check if the goal is dominated by any goal in the set."""
        for competitor in goal_set:
            if self.is_dominated(goal, competitor):
                return True, competitor
        return False, None
    
    def is_dominated(self, p, q):
        """Check if suggested goal p is dominated by goal q (assuming minimization or maximization)."""
        for idx in range(len(self.specs_id)):
            if self.specs_id[idx] == 'phm_min': #phm_min's desired spec range is a singleton, not a range
                continue
            if self.specs_id[idx][-3:] == 'max':
                if p[idx] < q[idx]:
                    return False
            else:
                if p[idx] > q[idx]:
                    return False
        return True
    
    def reset_idx(self):
        self.obj_idx=0
    
    def get_goals_norm(self):
        return self.goals_norm
    
    @abstractmethod
    def init_params(self):
        pass
    
    @abstractmethod
    def update_params(self,action):
        pass
    
    @abstractmethod
    def translate_params(self, cur_params):
        pass
