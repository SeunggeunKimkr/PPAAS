import numpy as np
import os
import scipy.interpolate as interp
import scipy.optimize as sciopt
import yaml
import importlib
import time
import math
import pdb

debug = False

from eval_engines.ngspice.ngspice_wrapper_parallel import NgSpiceWrapper

def ldo_dc(output_path, file_name):
        try:
            dc_fname = os.path.join(output_path, file_name)
            LDO_testbench_dc = open(dc_fname, 'r')
            lines_dc = LDO_testbench_dc.readlines()
            Vin_dc = []                    
            Vout_dc = []
            for line in lines_dc:
                Vdc = line.split(' ')
                Vdc = [i for i in Vdc if i != '']
                Vin_dc.append(float(Vdc[0]))
                Vout_dc.append(float(Vdc[1]))             
            return Vin_dc, Vout_dc
        except:
            print("Simulation errors, no .OP simulation results.")
            pdb.set_trace()
def ldo_tran(output_path, file_name):
        # Transient analysis result parser
        try:
            tran_fname = os.path.join(output_path,file_name)
            ldo_tb_tran = open(tran_fname, 'r')
            lines_tran = ldo_tb_tran.readlines()
            time = []
            Vout_tran = []
            for line in lines_tran:
                line = line.split(' ')
                line = [i for i in line if i != '']
                time.append(float(line[0]))
                Vout_tran.append(float(line[1])) 
            
            return time, Vout_tran
        except:
                print("Simulation errors, no .TRAN simulation results.")

def ldo_ac(output_path, file_name):
        try:
            ac_fname = os.path.join(output_path, file_name)
            LDO_testbench_ac = open(ac_fname, 'r')  
            lines_ac = LDO_testbench_ac.readlines()   
            freq = []
            Vout_mag = []
            Vout_ph = []
            for line in lines_ac:
                Vac = line.split(' ')
                Vac = [i for i in Vac if i != '']
                freq.append(float(Vac[0]))
                Vout_mag.append(float(Vac[1]))
                Vout_ph.append(float(Vac[3]))
            return freq, Vout_mag, Vout_ph             
        except:
            print("Simulation errors, no .AC simulation results.")
            
def ldo_lr_power_vos(output_path, file_name):
    try:
        LR_Power_vos_fname = os.path.join(output_path, file_name)
        LDO_testbench_LR_Power_vos = open(LR_Power_vos_fname, 'r')
        lines_dc = LDO_testbench_LR_Power_vos.readlines()
        IL = []                    
        LDR = []
        Power_maxload = []
        Power_minload = []
        vos_maxload = []
        vos_minload = []
        for line in lines_dc:
            Vdc = line.split(' ')
            Vdc = [i for i in Vdc if i != '']
            IL.append(float(Vdc[0]))
            LDR.append(float(Vdc[1])) 
            Power_maxload.append(float(Vdc[3])) 
            Power_minload.append(float(Vdc[5])) 
            vos_maxload.append(float(Vdc[7]))
            vos_minload.append(float(Vdc[9]))    

        return IL, LDR, Power_maxload, Power_minload, vos_maxload, vos_minload
    except:
        print("Simulation errors, no .OP simulation results.")

def extract_tran_data(path):
    # time_points = []
    # raw_data = []
    vin_data = []
    vout_data = []
    time_data = []

    raw_tran_data =  np.genfromtxt(path, skip_header=1)
    time_data = raw_tran_data[:, 0]
    vin_data = raw_tran_data[:, 1]
    vout_data = raw_tran_data[:, 3]
    return time_data, vin_data, vout_data

def analyze_amplifier_performance(vinp, vout, time, d0):
    vinp = np.array(vinp)  # Convert list to NumPy array
    vout = np.array(vout)
    time = np.array(time)

    # Function to extract step parameters from input signal
    def get_step_parameters(vinp, time):
        dv = np.diff(vinp)  # Calculate the difference between consecutive elements
        t0 = time[np.where(dv > 0)[0][0]]  # Detect rising edge time
        t1 = time[np.where(dv < 0)[0][0]]  # Detect falling edge time
        v0 = np.median(vinp[time < t0])  # Initial value before the step
        v1 = np.median(vinp[(time > t0) & (time < t1)])  # Value during the step
        return v0, v1, t0, t1

    # Extract step parameters from the input sequence
    v0, v1, t0, t1 = get_step_parameters(vinp, time)

    # Check amplifier stability before the step occurs
    pre_step_data = vout[time < t0]  # Data before the step
    delta0 = (pre_step_data - v0) / v0  # Calculate the percentage difference
    d0_settle = np.mean(np.abs(delta0))  # Calculate the average absolute difference
    stable = not np.any(np.abs(delta0) > d0)  # Check if amplifier is stable

    # Function to find the first index where the signal has settled
    def find_settling_time_index(delta, d0):
        for i in range(len(delta)):
            if np.all(np.abs(delta[i:]) < d0):  # Check if all subsequent values are within the threshold
                return i
        return None

    
    
    def get_slope_and_settling_time(vout, time, v0, v1, start_t, end_t, d0, mode):
        # Select data within the specified time range
        idx = (time >= start_t) & (time <= end_t)
        vout_segment = vout[idx]
        time_segment = time[idx]
    
        # Calculate the slope (Slew Rate, SR)
        target_value = v0 + (v1 - v0) / 2  # The midpoint between v0 and v1
        idx_target = np.where(vout_segment >= target_value)[0][0] if np.any(vout_segment >= target_value) else None
    
        # Calculate settling time
        if mode == 'positive':
            delta = (vout_segment - v1) / v1  # Calculate the percentage deviation for the positive step
        else:
            delta = (vout_segment - v0) / v0  # Calculate the percentage deviation for the negative step
    
        # Find the first index where the signal settles (all subsequent deltas are below the threshold d0)
        idx_settle = find_settling_time_index(delta, d0)
        if idx_settle is None:
            settling_time = np.nan  # If the signal does not settle, return NaN
            d_settle = np.mean(np.abs(delta))  # Calculate the mean deviation
        else:
            settling_time = time_segment[idx_settle] - start_t  # Calculate the settling time
            d_settle = np.mean(np.abs(delta[idx_settle:]))  # Calculate the mean deviation after settling
        
        SR=0.0
        return SR, settling_time, d_settle

    
    
    # Calculate the slope and settling time for the rising edge
    SR_p, settling_time_p, d1_settle = get_slope_and_settling_time(vout, time, v0, v1, t0, t1, d0, 'positive')

    # Calculate the slope and settling time for the falling edge
    SR_n, settling_time_n, d2_settle = get_slope_and_settling_time(vout, time, v0, v1, t1, np.max(time), d0, 'negative')

    # Return all calculated metrics including stability, slope, and settling times for both edges
    return d0_settle, d1_settle, d2_settle, stable, SR_p, settling_time_p, SR_n, settling_time_n



class CircuitClass(NgSpiceWrapper):
    def __init__(self, yaml_path, path, num_process=None, root_dir=None, design_netlists=None):
        super().__init__(yaml_path, path, num_process, root_dir, design_netlists)
        yaml_name = yaml_path.split('/')[-1].split('.')[0]
        self.comparator = True if 'comparator' in yaml_name else False
        self.ldo = True if 'ldo' in yaml_name else False

    def translate_result(self, output_path):
        """
        :param output_path:
        :return
            result: dict(spec_kwds, spec_value)
        """
        
        if self.comparator:
            noise_freq = 30e6
            delay, power = self.find_inoise(output_path, noise_freq)
            if power < 0:
                power = 150
            else:
                power = power * 100.0
            if delay > 1000:
                delay = 1000
            spec = dict(
                power=power,
                delay=delay
            )
        elif self.ldo:
            self.Vdd=2.0
            self.PSRR_1kHz = 1e3 #  from DC to 1kHz
            self.PSRR_10kHz=1e4
            self.PSRR_1MHz=1e6
            self.dc_results = ldo_dc(output_path=output_path, file_name='ldo_tb_dc.txt') #todo
            idx = int(self.Vdd/0.01 - 1/0.01) # since I sweep Vdc from 1V - 3V to avoid some bad DC points 
            self.Vdrop =  abs(self.Vdd - self.dc_results[1][idx])
            
            # _, self.load_reg = ldo_tran(output_path=output_path, file_name='ldo_tb_load_reg.txt')
            # idx_1 = int(len(self.load_reg)/4)
            # idx_2 = len(self.load_reg) - 1
            # self.Vload_reg_delta = abs(self.load_reg[idx_2] - self.load_reg[idx_1])
            
            self.psrr_results_maxload = ldo_ac(output_path=output_path, file_name='ldo_tb_psrr_maxload.txt')
            freq = self.psrr_results_maxload[0]
            # @ 10 kHz
            idx_10kHz = int(10 * np.log10(self.PSRR_10kHz))
            # @ 1 MHz
            idx_1MHz = int(10 * np.log10(self.PSRR_1MHz))
            self.PSRR_maxload_worst_10kHz = max(self.psrr_results_maxload[1][:idx_10kHz]) # in linear scale
            self.PSRR_maxload_worst_1MHz = max(self.psrr_results_maxload[1][:idx_1MHz]) # in linear scale
            self.PSRR_maxload_worst_above_1MHz = max(self.psrr_results_maxload[1][idx_1MHz:]) # in linear scale
        
            ''' PSRR performance at min load current '''
            self.psrr_results_minload = ldo_ac(output_path=output_path, file_name='ldo_tb_psrr_minload.txt')
            freq = self.psrr_results_minload[0]

            # @ 10 kHz
            idx_10kHz = int(10 * np.log10(self.PSRR_10kHz))
            # @ 1 MHz
            idx_1MHz = int(10 * np.log10(self.PSRR_1MHz))
            self.PSRR_minload_worst_10kHz = max(self.psrr_results_minload[1][:idx_10kHz]) # in linear scale
            self.PSRR_minload_worst_1MHz = max(self.psrr_results_minload[1][:idx_1MHz]) # in linear scale
            self.PSRR_minload_worst_above_1MHz = max(self.psrr_results_minload[1][idx_1MHz:]) # in linear scale
    
            
            ''' Loop-gain phase margin at max load current'''
            self.loop_gain_results_maxload = ldo_ac(output_path=output_path, file_name='ldo_tb_loop_gain_maxload.txt')
            freq = self.loop_gain_results_maxload[0]
            self.loop_gain_mag_maxload = 20*np.log10(self.loop_gain_results_maxload[1])
            self.loop_gain_phase_maxload = self.loop_gain_results_maxload[2] # in degree
            if self.loop_gain_mag_maxload[0] < 0: # if DC gain is smaller than 0 dB
                self.phase_margin_maxload = 0 # phase margin becomes meaningless 
            else:  
                try:
                    idx = [i for i,j in enumerate(self.loop_gain_mag_maxload[:-1] * self.loop_gain_mag_maxload[1:]) if j<0][0]+1 
                    phase_margin_maxload = np.min(self.loop_gain_phase_maxload[:idx]) + 180
                except: # this rarely happens: unity gain is larger than the frequency sweep
                    idx = len(self.loop_gain_phase_maxload)
                    phase_margin_maxload = np.min(self.loop_gain_phase_maxload[:idx]) + 180
                if phase_margin_maxload > 180 or phase_margin_maxload < 0:
                    self.phase_margin_maxload = 0
                else:
                    self.phase_margin_maxload = phase_margin_maxload
            
            ''' Loop-gain phase margin at min load current'''
            self.loop_gain_results_minload = ldo_ac(output_path=output_path, file_name='ldo_tb_loop_gain_minload.txt')
            freq = self.loop_gain_results_minload[0]
            self.loop_gain_mag_minload = 20*np.log10(self.loop_gain_results_minload[1])
            self.loop_gain_phase_minload = self.loop_gain_results_minload[2] # in degree
            if self.loop_gain_mag_minload[0] < 0: # if DC gain is smaller than 0 dB
                self.phase_margin_minload = 0 # phase margin becomes meaningless 
            else:  
                try:
                    idx = [i for i,j in enumerate(self.loop_gain_mag_minload[:-1] * self.loop_gain_mag_minload[1:]) if j<0][0]+1 
                    phase_margin_minload = np.min(self.loop_gain_phase_minload[:idx]) + 180
                except: # this rarely happens: unity gain is larger than the frequency sweep
                    idx = len(self.loop_gain_phase_minload)
                    phase_margin_minload = np.min(self.loop_gain_phase_minload[:idx]) + 180
                if phase_margin_minload > 180 or phase_margin_minload < 0:
                    self.phase_margin_minload = 0
                else:
                    self.phase_margin_minload = phase_margin_minload
            # pdb.set_trace()
            return {
            'Vdrop': self.Vdrop*1e3,
            # 'Vload_reg_delta': self.Vload_reg_delta*1e3,
            'mPSRR_maxload_worst_10kHz': -20*np.log10(self.PSRR_maxload_worst_10kHz),
            'mPSRR_maxload_worst_1MHz': -20*np.log10(self.PSRR_maxload_worst_1MHz),
            'mPSRR_maxload_worst_above_1MHz': -20*np.log10(self.PSRR_maxload_worst_above_1MHz),
            
            'mPSRR_minload_worst_10kHz': -20*np.log10(self.PSRR_minload_worst_10kHz),
            'mPSRR_minload_worst_1MHz': -20*np.log10(self.PSRR_minload_worst_1MHz),
            'mPSRR_minload_worst_above_1MHz': -20*np.log10(self.PSRR_minload_worst_above_1MHz),
            
            'phm_maxload': self.phase_margin_maxload, 
            'phm_minload': self.phase_margin_minload
            }
            
        else:
            # use parse output here
            freq, vout,  ibias, dc_sweep_interval, dc_sweep_output = self.parse_output(output_path)
            # gain = self.find_ac_gain(vout)
            gain, max_swing, min_swing = self.find_dc_vals(dc_sweep_interval, dc_sweep_output)
            ugbw = self.find_ugbw(freq, vout)
            phm = self.find_phm(freq, vout)
            voltage_swing = np.abs(max_swing - min_swing)
            tran_path = os.path.join(output_path,"tran.csv")
            tran_meas = self.get_tran_stable_meas(tran_path)
            t_settle = tran_meas["settlingTime"]
            sr_settle = tran_meas["SR"]


            spec = dict(
                ugbw=ugbw,
                gain=gain,
                phm=phm,
                ibias=ibias,
                vswing=voltage_swing,
                t_settle=t_settle
                # max_swing=max_swing,
                # min_swing=min_swing,
            )
            
        return spec
    
    #comparator
    def find_inoise(self, output_path, noise_freq):
        noise_fname = os.path.join(output_path, "tran.csv")
        if not os.path.isfile(noise_fname):
            inoise = None
        else:
            noise_raw_outputs = np.genfromtxt(noise_fname, skip_header=0)
            delay = noise_raw_outputs[1] / (1e-12)
            power = -1 * noise_raw_outputs[3] / (1e-12)
        return delay, power
    
    
    #opamps
    #aggregate the opamp results
    def parse_output(self, output_path):

        ac_fname = os.path.join(output_path, 'ac.csv')
        dc_fname = os.path.join(output_path, 'dc.csv')
        dc_sweep_fname = os.path.join(output_path, 'dc_sweep.csv')
        

        if not os.path.isfile(ac_fname) or not os.path.isfile(dc_fname) or not os.path.isfile(dc_sweep_fname):
            print("ac/dc/dc_sweep file doesn't exist: %s" % output_path)

        ac_raw_outputs = np.genfromtxt(ac_fname, skip_header=1)
        dc_raw_outputs = np.genfromtxt(dc_fname, skip_header=1)
        dc_sweeep_raw_outputs = np.genfromtxt(dc_sweep_fname, skip_header=1)
        freq = ac_raw_outputs[:, 0]
        vout_real = ac_raw_outputs[:, 1]
        vout_imag = ac_raw_outputs[:, 2]
        vout = vout_real + 1j*vout_imag
        ibias = -dc_raw_outputs[1]
        dc_sweep_interval = dc_sweeep_raw_outputs[1:, 0] - dc_sweeep_raw_outputs[:-1, 0]
        dc_sweep_vout = dc_sweeep_raw_outputs[:, 1]

        return freq, vout, ibias, dc_sweep_interval, dc_sweep_vout
    
    def find_ac_gain (self, vout):
        return np.abs(vout)[0]
    
    def find_dc_vals(self, dc_sweep_interval, dc_sweep_output):
        output_interval = dc_sweep_output[1:] - dc_sweep_output[:-1]
        gains = output_interval/ dc_sweep_interval
        gain = np.max(np.abs(gains))
        gain_idx = np.argmax(np.abs(gains))
        gain_th = gain / np.sqrt(2)
        swing_range_idx = np.where(np.abs(gains) > gain_th)[0]
        if len(swing_range_idx) == 0:
            vcm = dc_sweep_output[gain_idx]
            return gain, vcm, vcm
        else:
            min_swing = dc_sweep_output[swing_range_idx[0]]
            max_swing = dc_sweep_output[swing_range_idx[-1]]
        return gain, max_swing, min_swing
        

    def find_ugbw(self, freq, vout):
        gain = np.abs(vout)
        ugbw, valid = self._get_best_crossing(freq, gain, val=1)
        if valid:
            return ugbw
        else:
            return freq[0]

    def find_phm(self, freq, vout):
        gain = np.abs(vout)
        phase = np.angle(vout, deg=False)
        phase = np.unwrap(phase) # unwrap the discontinuity
        phase = np.rad2deg(phase) # convert to degrees
        #
        # plt.subplot(211)
        # plt.plot(np.log10(freq[:200]), 20*np.log10(gain[:200]))
        # plt.subplot(212)
        # plt.plot(np.log10(freq[:200]), phase)

        phase_fun = interp.interp1d(freq, phase, kind='quadratic')
        ugbw, valid = self._get_best_crossing(freq, gain, val=1)
        if valid:
            if phase_fun(ugbw) > 0:
                return -180+phase_fun(ugbw)
            else:
                return 180 + phase_fun(ugbw)
        else:
            return -180

    def get_tran_stable_meas(self, path):
        meas = {}
        d0 = 0.01  # Settling threshold
        time_data, vin_data, vout_data = extract_tran_data(path)  # Extract transient data
        if time_data is None:
            return None

        # Analyze amplifier performance and extract stability and slope metrics
        d0_settle, d1_settle, d2_settle, stable, SR_p, settling_time_p, SR_n, settling_time_n = analyze_amplifier_performance(vin_data, vout_data, time_data, d0)
        # print(d0_settle, d1_settle, d2_settle, stable, SR_p, settling_time_p, SR_n, settling_time_n)

        # Take the absolute values of the settling and slope values
        d0_settle = abs(d0_settle)
        d1_settle = abs(d1_settle)
        d2_settle = abs(d2_settle)
        SR_n = abs(SR_n)
        SR_p = abs(SR_p)
        settlingTime_p = abs(settling_time_p)
        settlingTime_n = abs(settling_time_n)

        # Handle NaN values for d0_settle
        if math.isnan(d0_settle):
            d0_settle = 10  # Assign a penalty value if d0_settle is NaN

        # Handle NaN values for d1_settle and d2_settle
        if math.isnan(d1_settle) or math.isnan(d2_settle):
            if math.isnan(d1_settle):
                d0_settle += 10  # Apply penalty if d1_settle is NaN
            if math.isnan(d2_settle):
                d0_settle += 10  # Apply penalty if d2_settle is NaN
            d_settle = d0_settle
        else:
            d_settle = max(d0_settle, d1_settle, d2_settle)  # Take the maximum settling value

        # Handle NaN values for SR (Slew Rate)
        if math.isnan(SR_p) or math.isnan(SR_n):
            SR = -d_settle  # Assign penalty if SR is not available
        else:
            SR = min(SR_p, SR_n)  # Take the minimum SR value

        # Handle NaN values for settling times
        if math.isnan(settlingTime_p) or math.isnan(settlingTime_n):
            settlingTime = d_settle  # Assign penalty if settling time is NaN
        else:
            settlingTime = max(settlingTime_p, settlingTime_n)  # Take the maximum settling time

        # Store the calculated metrics in the `meas` dictionary
        meas['d_settle'] = d_settle
        meas['SR'] = SR
        meas['settlingTime'] = settlingTime
        # print(meas['settlingTime'])
        return meas

    def _get_best_crossing(cls, xvec, yvec, val):
        interp_fun = interp.InterpolatedUnivariateSpline(xvec, yvec)

        def fzero(x):
            return interp_fun(x) - val

        xstart, xstop = xvec[0], xvec[-1]
        try:
            return sciopt.brentq(fzero, xstart, xstop), True
        except ValueError:
            # avoid no solution
            # if abs(fzero(xstart)) < abs(fzero(xstop)):
            #     return xstart
            return xstop, False
    

class TwoStageMeasManager(object):

    def __init__(self, design_specs_fname):
        self.design_specs_fname = design_specs_fname
        with open(design_specs_fname, 'r') as f:
            self.ver_specs = yaml.load(f)

        self.spec_range = self.ver_specs['spec_range']
        self.params = self.ver_specs['params']

        self.params_vec = {}
        self.search_space_size = 1
        for key, value in self.params.items():
            if value is not None:
                # self.params_vec contains keys of the main parameters and the corresponding search vector for each
                self.params_vec[key] = np.arange(value[0], value[1], value[2]).tolist()
                self.search_space_size = self.search_space_size * len(self.params_vec[key])

        self.measurement_specs = self.ver_specs['measurement']
        root_dir = self.measurement_specs['root_dir'] + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
        num_process = self.measurement_specs['num_process']

        self.netlist_module_dict = {}
        for netlist_kwrd, netlist_val in self.measurement_specs['netlists'].items():
            netlist_module = importlib.import_module(netlist_val['wrapper_module'])
            netlist_cls = getattr(netlist_module, netlist_val['wrapper_class'])
            self.netlist_module_dict[netlist_kwrd] = netlist_cls(num_process=num_process,
                                                                 design_netlist=netlist_val['cir_path'],
                                                                 root_dir=root_dir)

    def evaluate(self, design):
        state_dict = dict()
        for i, key in enumerate(self.params_vec.keys()):
            state_dict[key] = self.params_vec[key][design[i]]
        state = [state_dict]
        dsn_names = [design.id]
        results = {}
        for netlist_name, netlist_module in self.netlist_module_dict.items():
            results[netlist_name] = netlist_module.run(state, dsn_names)

        specs_dict = self._get_specs(results)
        specs_dict['cost'] = self.cost_fun(specs_dict)
        return specs_dict

    def _get_specs(self, results_dict):
        fdbck = self.measurement_specs['tb_params']['feedback_factor']
        tot_err = self.measurement_specs['tb_params']['tot_err']

        ugbw_cur = results_dict['ol'][0][1]['ugbw']
        gain_cur = results_dict['ol'][0][1]['gain']
        phm_cur = results_dict['ol'][0][1]['phm']
        ibias_cur = results_dict['ol'][0][1]['Ibias']

        # common mode gain and cmrr
        cm_gain_cur = results_dict['cm'][0][1]['cm_gain']
        cmrr_cur = 20 * np.log10(gain_cur / cm_gain_cur)  # in db
        # power supply gain and psrr
        ps_gain_cur = results_dict['ps'][0][1]['ps_gain']
        psrr_cur = 20 * np.log10(gain_cur / ps_gain_cur)  # in db

        # transient settling time and offset calculation
        t = results_dict['tran'][0][1]['time']
        vout = results_dict['tran'][0][1]['vout']
        vin = results_dict['tran'][0][1]['vin']

        tset_cur = self.netlist_module_dict['tran'].get_tset(t, vout, vin, fdbck, tot_err=tot_err)
        offset_curr = abs(vout[0] - vin[0] / fdbck)

        specs_dict = dict(
            gain=gain_cur,
            ugbw=ugbw_cur,
            pm=phm_cur,
            ibias=ibias_cur,
            cmrr=cmrr_cur,
            psrr=psrr_cur,
            offset_sys=offset_curr,
            tset=tset_cur,
        )

        return specs_dict

    def compute_penalty(self, spec_nums, spec_kwrd):
        if type(spec_nums) is not list:
            spec_nums = [spec_nums]
        penalties = []
        for spec_num in spec_nums:
            penalty = 0
            spec_min, spec_max, w = self.spec_range[spec_kwrd]
            if spec_max is not None:
                if spec_num > spec_max:
                    # penalty += w*abs((spec_num - spec_max) / (spec_num + spec_max))
                    penalty += w * abs(spec_num - spec_max) / abs(spec_num)
            if spec_min is not None:
                if spec_num < spec_min:
                    # penalty += w*abs((spec_num - spec_min) / (spec_num + spec_min))
                    penalty += w * abs(spec_num - spec_min) / abs(spec_min)
            penalties.append(penalty)
        return penalties

    def cost_fun(self, specs_dict):
        """
        :param design: a list containing relative indices according to yaml file
        :param verbose:
        :return:
        """
        cost = 0
        for spec in self.spec_range.keys():
            penalty = self.compute_penalty(specs_dict[spec], spec)[0]
            cost += penalty

        return cost
