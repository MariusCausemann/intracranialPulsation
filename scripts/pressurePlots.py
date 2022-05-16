from calendar import c
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import sys
plt.style.use('bmh') 
import matplotlib as mpl
mpl.rcParams['axes.facecolor'] = 'white' 
mpl.rcParams['lines.linewidth'] = 3


figsize = (6,4)
porous_id = 1
fluid_id = 2

mmHg2Pa = 132.32
m3tomL = 1e6

if __name__=="__main__":
    mesh_name = sys.argv[1]
    sim_name = sys.argv[2]

    sim_config_file = f"results/{mesh_name}_{sim_name}/config.yml"
    mesh_config_file = f"meshes/{mesh_name}/{mesh_name}_config.yml"
    plot_dir = f"results/{mesh_name}_{sim_name}/pressure_plots"
    key_quantities_file = f"results/{mesh_name}_{sim_name}/pressure_key_quantities.yml"
    timseries_file = f"results/{mesh_name}_{sim_name}/pressure_timeseries.yml"

    try:
        os.mkdir(plot_dir)
    except:
        pass

    with open(mesh_config_file) as conf_file:
        mesh_config = yaml.load(conf_file, Loader=yaml.FullLoader)
    with open(sim_config_file) as conf_file:
        sim_config = yaml.load(conf_file, Loader=yaml.FullLoader)
    with open(timseries_file) as conf_file:
        timeseries = yaml.load(conf_file, Loader=yaml.FullLoader)
    timeseries = {k:np.array(v) for k,v in timeseries.items()}

    T = sim_config["T"]
    num_steps = sim_config["num_steps"]

    dt = T/num_steps
    times = np.linspace(0, T, num_steps + 1)
    start_time = T - 1
    start_idx = np.where(times==start_time)[0][0] 
    times = times[start_idx:] - start_time
    probes = mesh_config["probes"]
    flatprobes = dict(**probes["sas"],**probes["parenchyma"],**probes["ventricular_system"])

    key_quantities = {}

    key_quantities["max_lat_ventricle_pressure"] = timeseries["lateral_ventricles"].max()/mmHg2Pa
    key_quantities["min_lat_ventricle_pressure"] = timeseries["lateral_ventricles"].min()/mmHg2Pa
    key_quantities["mean_lat_ventricle_pressure"] = timeseries["lateral_ventricles"].mean()/mmHg2Pa

    #plot pressure over time:

    plt.figure(figsize=figsize)
    plt.plot(times, timeseries["lateral_ventricles"]/mmHg2Pa, "-", label="LV")
    plt.plot(times, timeseries["fourth_ventricle"]/mmHg2Pa, '--', label="V4")
    plt.plot(times, timeseries["top_sas"]/mmHg2Pa, ":",color="black",  label="SAS")

    plt.legend()
    plt.xlabel("time [s]")
    plt.ylabel(" pressure [mmHg]")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/pressure_evolution.pdf")


    # plot pressure gradient over time
    
    LV_point = flatprobes["lateral_ventricles"]
    top_sas_point = flatprobes["top_sas"]
    v4_point = flatprobes["fourth_ventricle"]

    v4_dist = np.array(LV_point) - np.array(v4_point)
    v4_dist = np.linalg.norm(v4_dist)
    sas_dist = np.array(LV_point) - np.array(top_sas_point)
    sas_dist = np.linalg.norm(sas_dist)
    v4_diff = timeseries["lateral_ventricles"] - timeseries["fourth_ventricle"]
    sas_diff = timeseries["lateral_ventricles"] - timeseries["top_sas"]

    sas_grad = sas_diff/mmHg2Pa/sas_dist
    v4_grad = v4_diff/mmHg2Pa/v4_dist

    plt.figure(figsize=figsize)
    plt.plot(times, timeseries["lateral_ventricles"]*0.0, " ")
    plt.plot(times, v4_grad, '--',label="V4")
    plt.plot(times, sas_grad, ':', color="black", label="SAS")
    plt.legend()
    plt.xlabel("time [s]")
    plt.ylabel("pressure gradient [mmHg/m]")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/pressure_gradient.pdf")

    
    key_quantities["max_pressure_gradient"] = sas_grad.max()
    key_quantities["min_pressure_gradient"] = sas_grad.min()
    key_quantities["mean_pressure_gradient"] = sas_grad.mean()
    key_quantities["peak_pressure_gradient"] = abs(sas_grad).max()
    key_quantities["pressure_gradient_distance"] = float(sas_dist)
    
    # max parenchyml flow velocity

    max_flow_over_time = timeseries["max_flow_over_time"]
    mean_flow_over_time = timeseries["mean_flow_over_time"]

    key_quantities["max_parenchymal_flow"] = max_flow_over_time.max()
    key_quantities["mean_parenchymal_flow"] = mean_flow_over_time.max()

    plt.figure(figsize=figsize)
    plt.plot(times, max_flow_over_time*1e6, "-")
    plt.xlabel("time [s]")
    plt.ylabel("max flow [$\mu$m/s]")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/parenchymal_flow_over_time.pdf")

    key_quantities = {k: float(v) for k,v in key_quantities.items()} 

    with open(key_quantities_file, "w") as key_q_file:
        yaml.dump(key_quantities, key_q_file)



