from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import sys
from brainpulse.PostProcessing import (project_list, load_meshes)


plt.style.use('bmh') 
plt.rcParams['axes.facecolor'] = 'white' 
plt.rcParams['lines.linewidth'] = 3
figsize = (6, 4)
porous_id = 1
fluid_id = 2

mmHg2Pa = 132.32
m3tomL = 1e6

phases = [0.13, .35, 0.56, 0.8]

interface_id = 1
spinal_outlet_id = 3

names = ["d"]

if __name__=="__main__":
    mesh_name = sys.argv[1]
    sim_name = sys.argv[2]

    sim_config_file = f"results/{mesh_name}_{sim_name}/config.yml"
    mesh_config_file = f"meshes/{mesh_name}/{mesh_name}_config.yml"
    plot_dir = f"results/{mesh_name}_{sim_name}/flow_plots"
    try:
        os.mkdir(plot_dir)
    except:
        pass
    key_quantities_file = f"results/{mesh_name}_{sim_name}/displacement_key_quantities.yml"

    with open(mesh_config_file) as conf_file:
        mesh_config = yaml.load(conf_file, Loader=yaml.FullLoader)
    with open(sim_config_file) as conf_file:
        sim_config = yaml.load(conf_file, Loader=yaml.FullLoader)
    mesh, subdomain_marker, label_marker, boundary_marker, label_boundary_marker = load_meshes(mesh_name)
    gdim = mesh.geometric_dimension()
    sim_file = f"results/{mesh_name}_{sim_name}/results.hdf"
    source_conf = sim_config["source_data"]
    T = sim_config["T"]
    num_steps = sim_config["num_steps"]
    dt = T/num_steps
    times = np.linspace(0, T, num_steps + 1)
    probes = mesh_config["probes"]
    flatprobes = dict(**probes["sas"],**probes["parenchyma"],**probes["ventricular_system"])
    domains = mesh_config["domains"]
    name_to_label = {dom["name"]:dom["id"] for dom in domains}

    V = VectorFunctionSpace(mesh, "CG", 2)
    
    infile = HDF5File(mesh.mpi_comm(), sim_file, "r")

    key_quantities = {"num_cells":MPI.sum(MPI.comm_world, mesh.num_cells()),
                      "num_vertices":MPI.sum(MPI.comm_world, mesh.num_vertices()),
                      "hmax":MPI.max(MPI.comm_world, mesh.hmax()),
                      "hmin":MPI.min(MPI.comm_world, mesh.hmin()),
                     }
                      
    print(key_quantities)
    start_time = T -1 
    start_idx = np.where(times==start_time)[0][0] 

    results = {n:[] for n in names}
    times = times[start_idx:]

    for n in names:
        for i in range(num_steps + 1 - start_idx):
            warning(f"reading {n}, timestep {i}")
            f = Function(V)
            infile.read(f,f"{n}/vector_{i + start_idx}")
            results[n].append(f)
    infile.close()

    # displacement statistics

    V_abs = FunctionSpace(mesh, "CG", 1)

    d_abs = project_list([inner(d,d) for d in results["d"]], V_abs)

    max_disp_over_time = np.array([norm(d.vector(), "linf") for d in d_abs])
    max_disp_over_time = np.nan_to_num(max_disp_over_time)

    plt.figure(figsize=figsize)
    plt.plot(times - start_time, max_disp_over_time, "-*")
    plt.legend()
    plt.xlabel("time [s]")
    plt.ylabel("max displacement [m]")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/displ_over_time.pdf")

    key_quantities["max_displacement"] = max_disp_over_time.max()

    key_quantities = {k: float(v) for k,v in key_quantities.items()} 

    with open(key_quantities_file, "w") as key_q_file:
        yaml.dump(key_quantities, key_q_file)


    # compute parenchyma volume change
    ds_interf = Measure("dS", domain=mesh, subdomain_data=boundary_marker, subdomain_id=interface_id)
    dx = Measure("dx", domain=mesh, subdomain_data=subdomain_marker)
    n = FacetNormal(mesh)("-")
    par_V = np.array([assemble(dot(d("-"), n)*ds_interf + Constant(0.0)*dx) for d in results["d"]])
    par_dV = np.diff(par_V, append=par_V[0])/dt

    plt.figure(figsize=figsize)
    plt.plot(times, par_dV*m3tomL, "-")
    plt.xlabel("time [s]")
    plt.ylabel("dV [ml/s]")
    plt.savefig(f"{plot_dir}/parenchyma_vol_change.pdf")

    plt.figure(figsize=figsize)
    plt.plot(times, par_V*m3tomL, "-")
    plt.xlabel("time [s]")
    plt.ylabel("dV [ml/s]")
    plt.savefig(f"{plot_dir}/parenchyma_vol.pdf")
    


