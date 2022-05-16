from fenics import *
from multiphenics import *
import numpy as np
from brainpulse.PlottingHelper import extract_cross_section
import matplotlib.pyplot as plt
from brainpulse.meshExpression import meshExpression
from brainpulse.PostProcessing import (get_source_expression, project_list,
                                       load_meshes)
import sys
import yaml

plt.style.use('bmh') 
PETScOptions.set("mat_mumps_use_omp_threads", 4)
PETScOptions.set("mat_mumps_icntl_4", 3) # set mumps verbosity (0-4)
PETScOptions.set("mat_mumps_icntl_28", 2) # 2 for parallel analysis and ictnl(29) ordering
PETScOptions.set("mat_mumps_icntl_29", 2) # parallel ordering 1 = ptscotch, 2 = parmetis

porous_id = 1
fluid_id = 2

mmHg2Pa = 132.32
m3tomL = 1e6

parameters['ghost_mode'] = 'shared_vertex' 

names = ["pF", "pP", "phi"]

variables = {"pF":"fluid", "pP":"porous", "phi":"porous",
            "d":"porous", "u":"fluid"}

if __name__=="__main__":
    mesh_name = sys.argv[1]
    sim_name = sys.argv[2]

    sim_config_file = f"results/{mesh_name}_{sim_name}/config.yml"
    mesh_config_file = f"meshes/{mesh_name}/{mesh_name}_config.yml"
    timeseries_file = f"results/{mesh_name}_{sim_name}/pressure_timeseries.yml"

    with open(mesh_config_file) as conf_file:
        mesh_config = yaml.load(conf_file, Loader=yaml.FullLoader)
    with open(sim_config_file) as conf_file:
        sim_config = yaml.load(conf_file, Loader=yaml.FullLoader)
    mesh, subdomain_marker, label_marker, boundary_marker, label_boundary_marker = load_meshes(mesh_name)
    
    gdim = mesh.geometric_dimension()
    subd =  Function(FunctionSpace(mesh, "DG", 0))
    
    porous_dom = MeshFunction("size_t", mesh, dim=gdim)
    porous_dom.array()[:] = subdomain_marker.array()==1

    subd = CompiledExpression(meshExpression, dom=porous_dom, degree=0)

    sim_file = f"results/{mesh_name}_{sim_name}/results.hdf"
    source_conf = sim_config["source_data"]
    T = sim_config["T"]
    num_steps = sim_config["num_steps"]
    dt = T/num_steps
    times = np.linspace(0, T, num_steps + 1)
    source_expr, source_vals = get_source_expression(source_conf, mesh, subdomain_marker, porous_id, times)
    probes = mesh_config["probes"]
    flatprobes = dict(**probes["sas"],**probes["parenchyma"],**probes["ventricular_system"])

    infile = HDF5File(mesh.mpi_comm(), sim_file, "r")

    porous_restriction_file = f"meshes/{mesh_name}/{mesh_name}_porous.rtc.xdmf"
    fluid_restriction_file = f"meshes/{mesh_name}/{mesh_name}_fluid.rtc.xdmf"

    porousrestriction = MeshRestriction(mesh, porous_restriction_file)
    fluidrestriction = MeshRestriction(mesh, fluid_restriction_file)


    P1 = FunctionSpace(mesh, "CG", 1)

    H = BlockFunctionSpace([P1,P1,P1], restrict=[fluidrestriction, porousrestriction, porousrestriction])

    start_time = T - 1 
    start_idx = np.where(times==start_time)[0][0] 
    times = times[start_idx:]

    # read simulation data
    results = {n:[] for n in names}
    print("loading data...")
    for i in range(num_steps + 1 - start_idx):
        print(i)
        block_funcs = block_split(BlockFunction(H))
        for j, n in enumerate(names):
            f = block_funcs[j]
            infile.read(f,f"{n}/vector_{i + start_idx}")
            results[n].append(f)
    infile.close()

    # compute darcy flow
    results["flow"] = []
    DG = FunctionSpace(mesh, "DG", 0)

    kappa = sim_config["material_parameter"]["kappa"]
    mu_f = sim_config["material_parameter"]["mu_f"]
    results["flow_mag"] = project_list([sqrt(inner(kappa/mu_f * grad(pP), kappa/mu_f * grad(pP)))
                                        for pP in results["pP"]], DG)

    pressure_time_series = {}

    point_var_list = [("pF","top_sas"),
                      ("phi","top_parenchyma"),
                      ("pF","lateral_ventricles"),
                      ("pF","fourth_ventricle")]

    for (var, p_name) in point_var_list:
        data = extract_cross_section(results[var], [Point(flatprobes[p_name])]).flatten()
        pressure_time_series[p_name] = data

    # max parenchymal flow velocity

    max_flow_over_time = [norm(q.vector(), 'linf') for q in results["flow_mag"]]

    dxD = Measure("dx", domain=mesh, subdomain_data=subdomain_marker)
    v_par = assemble(1*dxD(porous_id))
    mean_flow_over_time = np.array([assemble(q*dxD(porous_id)) for q in results["flow_mag"]])
    mean_flow_over_time = (mean_flow_over_time) / v_par

    pressure_time_series["max_flow_over_time"] = max_flow_over_time
    pressure_time_series["mean_flow_over_time"] = mean_flow_over_time

    pressure_time_series = {k: np.array(v).tolist() for k,v in pressure_time_series.items()} 

    with open(timeseries_file, "w") as ts_file:
        yaml.dump(pressure_time_series, ts_file)



