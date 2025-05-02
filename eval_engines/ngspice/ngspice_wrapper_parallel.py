import re
import numpy as np
import copy
from multiprocessing.dummy import Pool as ThreadPool
import os
import abc
import scipy.interpolate as interp
import scipy.optimize as sciopt
import random
import time
import pprint
import yaml
import IPython
import shutil
import pdb
debug = False

class NgSpiceWrapper(object):

    BASE_TMP_DIR = os.path.abspath("./tmp/ckt_da")

    def __init__(self, yaml_path, path, num_process=None, root_dir=None, design_netlists = None):
        if root_dir == None:
            self.root_dir = NgSpiceWrapper.BASE_TMP_DIR
        else:
            self.root_dir = root_dir
        
        os.makedirs(self.root_dir, exist_ok=True)
        
        self.base_design_names = []
        self.gen_dirs = []
        self.tmp_lines = []
        if design_netlists is None:
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.full_load(f)
            design_netlists = yaml_data['dsn_netlist']
        
        self.num_process = len(design_netlists) if num_process == None else num_process
        
        for design_netlist in design_netlists:
            design_netlist = path+'/'+design_netlist
            _, dsg_netlist_fname = os.path.split(design_netlist)
            base_design_name = os.path.splitext(dsg_netlist_fname)[0]
            self.base_design_names.append(base_design_name)
            gen_dir = os.path.join(self.root_dir, "designs_" + base_design_name)
            self.gen_dirs.append(gen_dir)
            os.makedirs(gen_dir, exist_ok=True)
            raw_file = open(design_netlist, 'r')
            self.tmp_lines.append(raw_file.readlines())
            raw_file.close()

    def get_design_name(self, state, base_design_name):
        fname = base_design_name
        for value in state.values():
            round_value = np.round(float(value), 9)
            fname += "_" + str(round_value)
        return fname

    def create_design(self, state, new_fname, gen_dir, tmp_lines):
        design_folder = os.path.join(gen_dir, new_fname) + "_" + str(random.randint(0,1000000))
        os.makedirs(design_folder, exist_ok=True)

        fpath = os.path.join(design_folder, new_fname + '.cir')

        lines = copy.deepcopy(tmp_lines)
        for line_num, line in enumerate(lines):
            if '.include' in line:
                regex = re.compile("\.include\s*\"(.*?)\"")
                found = regex.search(line)
                if found:
                    pass # do not change the model path
            if '.param' in line:
                for key, value in state.items():
                    regex = re.compile("%s=(\S+)" % (key))
                    found = regex.search(line)
                    if found:
                        new_replacement = "%s=%s" % (key, str(value))
                        lines[line_num] = lines[line_num].replace(found.group(0), new_replacement)
            if 'wrdata' in line:
                regex = re.compile("wrdata\s*(\w+\.\w+)\s*")
                found = regex.search(line)
                if found:
                    replacement = os.path.join(design_folder, found.group(1))
                    lines[line_num] = lines[line_num].replace(found.group(1), replacement)
            if 'write' in line:
                regex = re.compile("write\s*(\w+\.\w+)\s*")
                found = regex.search(line)
                if found:
                    replacement = os.path.join(design_folder, found.group(1))
                    lines[line_num] = lines[line_num].replace(found.group(1), replacement)

        with open(fpath, 'w') as f:
            f.writelines(lines)
            f.close()
        return design_folder, fpath

    def simulate(self, fpath):
        info = 0 # this means no error occurred
        command = "ngspice -b %s >/dev/null 2>&1" %fpath
        exit_code = os.system(command)
        if debug:
            print(command)
            print(fpath)

        if (exit_code % 256):
            raise RuntimeError('program {} failed!'.format(command))
            info = 1 # this means an error has occurred
        return info

    def create_design_and_simulate(self, state, base_design_name, gen_dir, tmp_lines, verbose=False):
        if debug:
            print('state', state)
            print('verbose', verbose)
        dsn_name = self.get_design_name(state, base_design_name)
        if verbose:
            print(dsn_name)
        design_folder, fpath = self.create_design(state, dsn_name, gen_dir, tmp_lines)
        info = self.simulate(fpath)
        specs = self.translate_result(design_folder)
        shutil.rmtree(design_folder)
        return state, specs, info


    def run(self, state, design_names=None, verbose=False):
        """
        :param states:
        :param design_names: if None default design name will be used, otherwise the given design name will be used
        :param verbose: If True it will print the design name that was created
        :return:
            results = [(state: dict(param_kwds, param_value), specs: dict(spec_kwds, spec_value), info: int)]
        """
        if design_names == None:
            design_names =  self.base_design_names
        run_parallel = True
        if run_parallel:
            pool = ThreadPool(processes=self.num_process)
            arg_list = [(state, dsn_name, gen_dir, tmp_lines, verbose) for (dsn_name, gen_dir, tmp_lines)in zip(design_names, self.gen_dirs, self.tmp_lines)]
            results = pool.starmap(self.create_design_and_simulate, arg_list)
            pool.close()
            states, specs, infos = zip(*results)
        else:
            states = []
            specs = []
            infos = []
            for (dsn_name, gen_dir, tmp_lines) in zip(design_names, self.gen_dirs, self.tmp_lines):
                state, spec, info = self.create_design_and_simulate(state, dsn_name, gen_dir, tmp_lines, verbose)
                states.append(state)
                specs.append(spec)
                infos.append(info)
        return states, specs, infos

    def translate_result(self, output_path):
        """
        This method needs to be overwritten according to cicuit needs,
        parsing output, playing with the results to get a cost function, etc.
        The designer should look at his/her netlist and accordingly write this function.

        :param output_path:
        :return:
        """
        result = None
        return result
