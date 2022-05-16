import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import sys
plt.style.use('bmh') 
import matplotlib as mpl
mpl.rcParams['axes.facecolor'] = 'white' 
mpl.rcParams['lines.linewidth'] = 3

figsize = (6, 4)

mmHg2Pa = 132.32
m3tomL = 1e6

phases = [0.13, .35, 0.56, 0.8]

if __name__=="__main__":
    mesh_name = sys.argv[1]
    sim_name = sys.argv[2]

    sim_config_file = f"results/{mesh_name}_{sim_name}/config.yml"
    mesh_config_file = f"meshes/{mesh_name}/{mesh_name}_config.yml"
    plot_dir = f"results/{mesh_name}_{sim_name}/flow_plots"
    key_quantities_file = f"results/{mesh_name}_{sim_name}/flow_key_quantities.yml"
    timseries_file = f"results/{mesh_name}_{sim_name}/flow_timeseries.yml"

    try:
        os.mkdir(plot_dir)
    except:
        pass

    with open(mesh_config_file) as conf_file:
        mesh_config = yaml.load(conf_file, Loader=yaml.FullLoader)
    with open(sim_config_file) as conf_file:
        sim_config = yaml.load(conf_file, Loader=yaml.FullLoader)
    with open(timseries_file) as ts_file:
        timeseries = yaml.load(ts_file, Loader=yaml.FullLoader)
    timeseries = {k:np.nan_to_num(v) for k,v in timeseries.items()}

    T = sim_config["T"]
    num_steps = sim_config["num_steps"]
    dt = T/num_steps
    times = np.linspace(0, T, num_steps + 1)

    start_time = T -1
    start_idx = np.where(times==start_time)[0][0] 
    times = times[start_idx:] - start_time

    # plot source in inflow

    inflow = timeseries["inflow"]
    plt.figure(figsize=figsize)
    plt.plot(times, inflow[:len(times)]*m3tomL , "-")
    plt.xlabel("time [s]")
    plt.ylabel("inflow [ml/s]")
    plt.savefig(f"{plot_dir}/inflow.pdf")

    phases_end = np.where(times==1)[0][0] 
    plt.figure(figsize=figsize)
    plt.plot(times[:phases_end +1],inflow[:phases_end +1]*m3tomL , "-")
    plt.xlabel("time [s]")
    plt.ylabel("inflow in [ml/s]")
    #plt.title("net blood inflow")
    rom_phases = ["I", "II", "III", "IV"]
    for i,pt in enumerate(phases):
        plt.axvline(pt, c="red", ls=":")
        plt.annotate(f"({rom_phases[i]})", (pt - 0.05, 9.0))
    plt.xlim(0,1)
    plt.savefig(f"{plot_dir}/inflow_phases.pdf")

    key_quantities = {}
    key_quantities["max_blood_inflow"] = inflow.max()
    key_quantities["min_blood_inflow"] = inflow.min()
    key_quantities["mean_blood_inflow"] = inflow.mean()


    # plot outflow into spinal canal 
    
    spinal_outflow = timeseries["spinal_outflow"]
    plt.figure(figsize=figsize)
    plt.plot(times , spinal_outflow*m3tomL, "-")
    plt.xlabel("time [s]")
    plt.ylabel("flow rate [ml/s]")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/spinal_out_CSF.pdf")
    
    key_quantities["max_spinal_outflow"] = spinal_outflow.max()
    key_quantities["min_spinal_outflow"] = spinal_outflow.min()
    key_quantities["mean_spinal_outflow"] = spinal_outflow.mean()

    # cumulative outflow in spinal canal

    cum_outflow = np.cumsum(spinal_outflow)*dt

    key_quantities["max_cum_spinal_outflow"] = cum_outflow.max()
    key_quantities["min_cum_spinal_outflow"] = cum_outflow.min()
    key_quantities["mean_cum_spinal_outflow"] = cum_outflow.mean()

    plt.figure(figsize=figsize)
    plt.plot(times, cum_outflow*m3tomL, "-")
    plt.xlabel("time [s]")
    plt.ylabel("volume [ml]")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/cum_spinal_out_CSF.pdf")

    # ventricular flow

    plt.figure(figsize=figsize)

    flow_names = {"lateral_ventricles -> foramina":"LV -> FM",
                  "aqueduct -> fourth_ventricle":"AQ -> V4" ,
                  "fourth_ventricle -> median_aperture":"V4 -> MA"}

    for name, label in flow_names.items():
        flow = timeseries[name]
        plt.plot(times, flow * m3tomL, "-", label=label)
        key_quantities[f"max_flow_{name}"] = flow.max()


    plt.legend()
    plt.xlabel("time [s]")
    plt.ylabel("flow rate [ml/s]")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/ventr_CSF_flow.pdf")

    plt.figure(figsize=figsize)
    for name, label in flow_names.items():
        flow = timeseries[name]
        cum_flow = np.cumsum(flow)*dt - (np.cumsum(flow)*dt).min()
        plt.plot(times, np.cumsum(flow)*dt*m3tomL, "-", label=label)
        key_quantities[f"max_cum_flow_{name}"] = cum_flow.max()
        key_quantities[f"min_cum_flow_{name}"] = cum_flow.min()
        key_quantities[f"mean_cum_flow_{name}"] = cum_flow.mean()

    plt.legend()
    plt.xlabel("time [s]")
    plt.ylabel("volume [ml]")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/cum_CSF_flow.pdf")


    
    csf_interface_flow = timeseries["csf_interface_flow"]
    plt.figure(figsize=figsize)
    plt.plot(times , csf_interface_flow*m3tomL, "-")
    plt.xlabel("time [s]")
    plt.ylabel("flow rate [ml/s]")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/csf_interface_flow.pdf")
    
    key_quantities["max_csf_interface_flow"] = csf_interface_flow.max()
    key_quantities["min_csf_interface_flow"] = csf_interface_flow.min()
    key_quantities["mean_csf_interface_flow"] = csf_interface_flow.mean()

    key_quantities = {k: float(v) for k,v in key_quantities.items()} 
    with open(key_quantities_file, "w") as key_q_file:
        yaml.dump(key_quantities, key_q_file)
