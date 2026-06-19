import os
import sys
import time
import pickle
import brian2 as br
from brian2 import ms, mV, Hz, second, nS
import numpy as np
import argparse
from brian2.units import check_units
from brian2.core.functions import implementation


def main_loop(seed=30001, target_e=3, target_i=10):
    start_time = time.time()
    np.random.seed(seed)
    br.seed(seed)

    # Network parameters
    vr = -60*mV
    gL = 10*br.nS
    Ib = 200*br.pA
    tau_mem = 20*br.ms
    vI = -80*mV
    vE = 0*mV
    tau_E = 5*br.ms
    tau_I = 10*br.ms

    @implementation('cpp','''
    // Note that functions always need a return value at the moment
    double store_spike(int i, double t, bool when) {
    if (when)
    {static std::ofstream spike_file("spikes.txt", std::ios::app);
    spike_file << i << " " << t << std::endl;}
    return 0.;  // unused
    }
    ''')

    @check_units(i=1, t=br.second, when=1, result=1)
    def store_spike(i, t, when):
        raise NotImplementedError('Use standalone mode')

    save_times_s = [150, 290, 300, 450, 600] # in sec
    
    dtt = 10
    save_idx = [int(t / dtt) for t in save_times_s]
    Tmax = save_idx[-1]
    rec = [0] * (Tmax + 1)
    for d in save_idx:
        rec[d] = 1

    recwhen = br.TimedArray(rec, dt=10*br.second)

    #Neuron model
    eqs = '''
    dv/dt = (gL*(vr - v)+(gE*(vE-v)+gI*(vI-v)+Ib))/(gL*tau_mem) : volt (unless refractory)
    dgE/dt = -gE / tau_E : siemens
    dgI/dt = -gI / tau_I : siemens
    '''

    total_exc = int(total_neurons*4 / 5)
    total_inh = total_neurons - total_exc

    #Neuron group
    G = br.NeuronGroup(total_neurons, eqs, threshold='v>-50*mV', 
                       reset='v=vr; dummy_var = store_spike(i, t, int(recwhen(t)))', 
                       refractory=5*br.ms, 
                       method='euler') 
    SE = G[:total_exc]
    SI = G[total_exc:]
    
    @implementation('cpp', r"""
    #include <fstream>

    // Note that functions always need a return value at the moment
    double store_weights_IE(double w, double t, int when) {
        static std::ofstream weight_file("weights_IE.txt", std::ios::app);
        static double _prev_t = -1.;
        
        if (!when) {
            return 0.;
        }
        if (_prev_t != t) {
            if (_prev_t != -1.)
                weight_file << std::endl;
            weight_file << t << " ";
            _prev_t = t;
        }
        weight_file << w << " ";
        return 0.;  // unused
    }
    """)
    @check_units(w=1, t=br.second, when=1, result=1)
    def store_weights_IE(w, t, when):
        raise NotImplementedError('Use standalone mode')

    @implementation('cpp', r"""
    #include <fstream>

    // Note that functions always need a return value at the moment
    double store_weights_II(double w, double t, int when) {
        static std::ofstream weight_file("weights_II.txt", std::ios::app);  // opens the first time
        static double _prev_t = -1.;
        // Store all values for the same time in one line: "t w1 w2 w3 ..."
        if (!when) {
            return 0.;  // TimedArray says: don't record
        }
        if (_prev_t != t) {
            if (_prev_t != -1.)
                weight_file << std::endl;
            weight_file << t << " ";
            _prev_t = t;
        }
        weight_file << w << " ";
        return 0.;  // unused
    }
    """)
    @check_units(w=1, t=br.second, when=1, result=1)
    def store_weights_II(w, t, when):
        raise NotImplementedError('Use standalone mode')
    
    # Connections
    wII = 3*br.nS        
    wE = 0.3*br.nS       

    synEE = br.Synapses(SE, SE, 'w : siemens', on_pre='gE += w')
    synEE.connect(p=0.02, condition='i!=j')
    synEE.w = wE

    synEI = br.Synapses(SE, SI, 'w : siemens', on_pre='gE += w')
    synEI.connect(p=0.02, condition='i!=j')
    synEI.w = wE

    # synII = br.Synapses(SI, SI, 'w : siemens', on_pre='gI += wII') 
    # synII.connect(p=0.02, condition='i!=j')
    # synII.w = wII

    #Plasticity parameteres
    p0 = target_e * br.Hz      #target firing rate for excitatory neurons
    p0_I = target_i *br.Hz     #target firing rate for inhibitory neurons
    tau_STDP = 20*br.ms
    n = 1e-2                   #learning rate
    alpha = 2 * p0 * tau_STDP 
    alpha_I = 2 * p0_I * tau_STDP 
    w_min = 0
    w_max = 30

    #Plasticity model
    eqs_STDP='''
    w : 1
    dApre/dt = -Apre/tau_STDP : 1 (event-driven)
    dApost/dt = -Apost/tau_STDP : 1 (event-driven)
    '''

    synIE = br.Synapses(SI, SE, model=eqs_STDP,
                    on_pre='''
                    Apre += 1
                    w = clip(w + n * (Apost - alpha), w_min, w_max)
                    gI += w * nS
                    ''',
                    on_post='''
                    Apost += 1
                    w = clip(w + n * Apre, w_min, w_max)
                    ''' )
    synIE.connect(p=0.02, condition='i!=j')
    synIE.run_regularly('dummy_value = store_weights_IE(w, t, int(recwhen(t)))', dt=200*br.ms)
    synIE.w = 1e-10

    synII = br.Synapses(SI, SI, model=eqs_STDP,
                    on_pre='''
                    Apre += 1
                    w = clip(w + n * (Apost - alpha_I), w_min, w_max)
                    gI += w * nS
                    ''',
                    on_post='''
                    Apost += 1
                    w = clip(w + n * Apre, w_min, w_max)
                    ''' )
    synII.connect(p=0.02, condition='i!=j')
    synII.run_regularly('dummy_value = store_weights_II(w, t, int(recwhen(t)))', dt=200*br.ms)
    synII.w = 1e-10

    br.run(300*second)
    print('Here we would introduce an EI assembly and continue running')
    synEE.w['i<784 and j<784'] = 1.5*br.nS 
    synEI.w['i<784 and j<196'] = 1.5*br.nS 
    br.run(100*second)
    br.device.build()

    exc_offset = 0
    inh_offset = total_exc  

    # IE: pre in SI -> global = total_exc + i ; post in SE -> global = j
    pre_IE = inh_offset + np.asarray(synIE.i[:])
    post_IE = np.asarray(synIE.j[:])
    w_IE = np.asarray(synIE.w[:])

    # II: pre in SI -> global = total_exc + i ; post in SI -> global = total_exc + j
    pre_II = inh_offset + np.asarray(synII.i[:])
    post_II = inh_offset + np.asarray(synII.j[:])
    w_II = np.asarray(synII.w[:])

    np.savez(
        os.path.join(br.device.project_dir, "no_assembly_final_weights_and_connectivity_iall_400sec.npz"),
        pre_IE=pre_IE, post_IE=post_IE, w_IE=w_IE,
        pre_II=pre_II, post_II=post_II, w_II=w_II,
        total_neurons=total_neurons, total_exc=total_exc, total_inh=total_inh
    )
    print("Saved final weights + connectivity to no_assembly_final_weights_and_connectivity_iall_400sec.npz")
    print('Elapsed time', time.time() - start_time)

def single_run(seed=42, target_e=3, target_i=10):
    """ Function called in main, runs single simulation with given params.

        This function sets up the simulation environment, initializes parameters, 
        and calls `main_loop` to execute the simulation. Results are saved in the 
        `./output/` folder with a filename based on the parameters.

        Args:
            seed (float): Seed of the simulation.
            target_e (float): Target activity of the excitatory population in Hz for IE plasticity rule.
            target_i (float): Target activity of the inhibitory population in Hz for II plasticity rule.

        Returns:
            None
    """

    print('Performing a single run of the simulation')
    print(f'Dumps results in ./output/* folder')
    print(f'N neurons {total_neurons}; Sim. time {sim_time} seconds')
    br.device.reinit()
    br.device.activate()
    folder = f"./output/no_autapse_no_assembly_{seed}_{target_e}_{target_i}_5min"
    br.set_device('cpp_standalone', build_on_run=False, clean=True, directory=folder)
    # br.prefs.devices.cpp_standalone.openmp_threads = 1
    main_loop(seed, target_e, target_i)
    return

def main():
    parser = argparse.ArgumentParser(description="Run simulation with target activity and seed selection.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--target_e", type=float, required=True, help="Target excitatory rate")
    parser.add_argument("--target_i", type=float, required=True, help="Target inhibitory rate")
    args = parser.parse_args()

    print(f"Loading seed '{args.seed}' with target activity {args.target_e} Hz for E neurons, and {args.target_i} Hz for I neurons")

    single_run(seed=args.seed, target_e=args.target_e, target_i=args.target_i)

total_neurons = 10000
sim_time = 400*second

if __name__ == "__main__":
    main()