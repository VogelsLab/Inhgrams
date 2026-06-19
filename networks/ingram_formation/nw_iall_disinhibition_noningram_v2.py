import os
import time
import brian2 as br
from brian2 import ms, mV, Hz, second, nS
import numpy as np
import argparse
from brian2.units import check_units
from brian2.core.functions import implementation


def main_loop(seed=30001, target_e=3, target_i=10, seed_disinhibition=42,
              folder='output', n_neurons=1, repeat=0,
              groups_file="42_inhibitory_groups_topN.npz"):
    start_time = time.time()
    np.random.seed(seed)
    br.seed(seed)

    # Network parameters
    vr       = -60*mV
    gL       = 10*br.nS
    Ib       = 200*br.pA
    tau_mem  = 20*br.ms
    vI       = -80*mV
    vE       = 0*mV
    tau_E    = 5*br.ms
    tau_I    = 10*br.ms

    @implementation('cpp','''
    double store_spike(int i, double t, bool when) {
    if (when)
    {static std::ofstream spike_file("spikes.txt", std::ios::app);
    spike_file << i << " " << t << std::endl;}
    return 0.;
    }
    ''')
    @check_units(i=1, t=br.second, when=1, result=1)
    def store_spike(i, t, when):
        raise NotImplementedError('Use standalone mode')

    save_times_s = [590, 600, 610, 700]
    dtt          = 10
    save_idx     = [int(t / dtt) for t in save_times_s]
    Tmax         = save_idx[-1]
    rec          = [0] * (Tmax + 1)
    for d in save_idx:
        rec[d] = 1
    recwhen = br.TimedArray(rec, dt=10*br.second)

    # Neuron model
    eqs = '''
    dv/dt = (gL*(vr - v)+(gE*(vE-v)+gI*(vI-v)+Ib))/(gL*tau_mem) : volt (unless refractory)
    dgE/dt = -gE / tau_E : siemens
    dgI/dt = -gI / tau_I : siemens
    '''

    total_exc = int(total_neurons * 4 / 5)
    total_inh = total_neurons - total_exc

    # Neuron group
    G = br.NeuronGroup(total_neurons, eqs, threshold='v>-50*mV',
                       reset='v=vr; dummy_var = store_spike(i, t, int(recwhen(t)))',
                       refractory=5*br.ms,
                       method='euler')
    SE = G[:total_exc]
    SI = G[total_exc:]

    stimulus = br.TimedArray(
        np.append(np.append(np.zeros(600, dtype=int), 100), np.zeros(30, dtype=int)) * Hz,
        dt=1*second
    )
    P = br.PoissonGroup(1000, rates='stimulus(t)')

    groups       = np.load(groups_file)
    rest_abs_ids = groups["rest"]                        # absolute neuron IDs
    rest_local   = rest_abs_ids - total_exc              # to local inh indices
    rest_local   = rest_local[(rest_local >= 0) & (rest_local < total_inh)]

    # each repeat gets a different random draw
    rng_sample = np.random.default_rng(seed_disinhibition * 1000 + repeat)
    indices    = rng_sample.choice(rest_local, size=n_neurons, replace=False)
    indices    = np.asarray(indices, dtype=int)
    print(f"repeat={repeat} | sampling {n_neurons} neurons from REST group "
          f"(pool size {len(rest_local)})")
    print(f"Local inhibitory indices: {indices}")

    # Poisson to selected inhibitory neurons
    SynP   = br.Synapses(P, SI, 'w : siemens', on_pre='gI += w')
    p_conn = 0.05
    rng    = np.random.default_rng(seed_disinhibition)
    Npre   = len(P)
    M      = rng.random((Npre, len(indices))) < p_conn
    pre_idx, col = np.where(M)
    post_idx     = indices[col]
    SynP.connect(i=pre_idx, j=post_idx)
    SynP.w = 3 * nS

    # Fixed synapses
    wE = 0.3 * br.nS

    synEE = br.Synapses(SE, SE, 'w : siemens', on_pre='gE += w')
    synEE.connect(p=0.02, condition='i!=j')
    synEE.w = wE

    synEI = br.Synapses(SE, SI, 'w : siemens', on_pre='gE += w')
    synEI.connect(p=0.02, condition='i!=j')
    synEI.w = wE

    # plasticity params
    p0       = target_e * br.Hz
    p0_I     = target_i * br.Hz
    tau_STDP = 20 * br.ms
    eta      = 1e-2
    alpha    = 2 * p0   * tau_STDP
    alpha_I  = 2 * p0_I * tau_STDP
    w_min    = 0
    w_max    = 30

    eqs_STDP = '''
    w : 1
    dApre/dt  = -Apre  / tau_STDP : 1 (event-driven)
    dApost/dt = -Apost / tau_STDP : 1 (event-driven)
    '''

    synIE = br.Synapses(SI, SE, model=eqs_STDP,
                        on_pre='''
                        Apre += 1
                        w = clip(w + eta * (Apost - alpha), w_min, w_max)
                        gI += w * nS
                        ''',
                        on_post='''
                        Apost += 1
                        w = clip(w + eta * Apre, w_min, w_max)
                        ''')
    synIE.connect(p=0.02, condition='i!=j')
    synIE.w = 1e-10

    synII = br.Synapses(SI, SI, model=eqs_STDP,
                        on_pre='''
                        Apre += 1
                        w = clip(w + eta * (Apost - alpha_I), w_min, w_max)
                        gI += w * nS
                        ''',
                        on_post='''
                        Apost += 1
                        w = clip(w + eta * Apre, w_min, w_max)
                        ''')
    synII.connect(p=0.02, condition='i!=j')
    synII.w = 1e-10

    # run
    br.run(300 * second)
    print('Introducing engram and continuing...')
    synEE.w['i<784 and j<784'] = 1.5 * br.nS
    br.run(300 * second)
    eta = 0
    br.run(15 * second)
    br.device.build(directory=folder)
    print('Elapsed time', time.time() - start_time)

    np.savetxt(f"{folder}/inhibited_indices.txt", indices, fmt="%d")

    # save weights
    inh_offset = total_exc

    pre_EE  = np.asarray(synEE.i[:])
    post_EE = np.asarray(synEE.j[:])
    w_EE    = np.asarray(synEE.w[:])

    pre_EI  = np.asarray(synEI.i[:])
    post_EI = inh_offset + np.asarray(synEI.j[:])
    w_EI    = np.asarray(synEI.w[:])

    pre_IE  = inh_offset + np.asarray(synIE.i[:])
    post_IE = np.asarray(synIE.j[:])
    w_IE    = np.asarray(synIE.w[:])

    pre_II  = inh_offset + np.asarray(synII.i[:])
    post_II = inh_offset + np.asarray(synII.j[:])
    w_II    = np.asarray(synII.w[:])

    out_path = os.path.join(folder, f"weights_and_connectivity_rest_n{n_neurons}_r{repeat}.npz")
    np.savez(
        out_path,
        pre_EE=pre_EE, post_EE=post_EE, w_EE=w_EE,
        pre_EI=pre_EI, post_EI=post_EI, w_EI=w_EI,
        pre_IE=pre_IE, post_IE=post_IE, w_IE=w_IE,
        pre_II=pre_II, post_II=post_II, w_II=w_II,
        total_neurons=total_neurons, total_exc=total_exc, total_inh=total_inh
    )
    print(f"Saved weights to {out_path}")


def single_run(seed=42, target_e=3, target_i=10, seed_disinhibition=42,
               n_neurons=1, repeat=0,
               groups_file="42_inhibitory_groups_topN.npz"):
    print(f'Single run | N neurons: {total_neurons} | '
          f'n_inhibited: {n_neurons} | repeat: {repeat}')
    br.device.reinit()
    br.device.activate()
    folder = (f"./output/no_autapse_rest_n{n_neurons}_r{repeat}_ingram_iall_{seed}_{target_e}_{target_i}_{seed_disinhibition}_615sec")
    br.set_device('cpp_standalone', build_on_run=False, clean=True)
    main_loop(seed, target_e, target_i, seed_disinhibition,
              folder, n_neurons, repeat, groups_file)


def main():
    valid_n = [1, 2, 5, 10, 15, 20, 25, 50, 75, 100, 200]

    parser = argparse.ArgumentParser(
        description="Disinhibition control: sample N random neurons from REST group."
    )
    parser.add_argument("--seed",               type=int,   required=True)
    parser.add_argument("--target_e",           type=float, required=True)
    parser.add_argument("--target_i",           type=float, required=True)
    parser.add_argument("--seed_disinhibition", type=int,   required=True)
    parser.add_argument("--n_neurons",          type=int,   required=True,
                        choices=valid_n,
                        help=f"How many REST neurons to disinhibit. Options: {valid_n}")
    parser.add_argument("--repeat",             type=int,   required=True,
                        help="Repeat index 0–14 (each gets a different random sample)")
    parser.add_argument("--groups_file",        type=str,
                        default="42_inhibitory_groups_topN.npz",
                        help="Path to the saved inhibitory groups .npz file")
    args = parser.parse_args()

    print(f"Seed: {args.seed} | target_e: {args.target_e} Hz | "
          f"target_i: {args.target_i} Hz | n_neurons: {args.n_neurons} | "
          f"repeat: {args.repeat}")

    single_run(
        seed=args.seed,
        target_e=args.target_e,
        target_i=args.target_i,
        seed_disinhibition=args.seed_disinhibition,
        n_neurons=args.n_neurons,
        repeat=args.repeat,
        groups_file=args.groups_file
    )

total_neurons = 10000
sim_time      = 615 * second

if __name__ == "__main__":
    main()