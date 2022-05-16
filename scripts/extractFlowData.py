from fenics import *
import numpy as np
from brainpulse.PlottingHelper import extract_cross_section
import yaml
from brainpulse.PostProcessing import (get_source_expression,
                                           load_meshes)
import sys

parameters['ghost_mode'] = 'shared_facet' 

porous_id = 1
fluid_id = 2

interface_id = 1
spinal_outlet_id = 3

names = ["u"]

variables = {"pF":"fluid", "pP":"porous", "phi":"porous",
            "d":"porous", "u":"fluid"}

def compute_internal_flow(dom1, dom2):
    """ compute flow from dom1 into dom2"""
    dom1_id = name_to_label[dom1]
    dom2_id = name_to_label[dom2]
    intf_id = int(f"{min([dom1_id, dom2_id])}{max([dom1_id, dom2_id])}")
    ds_intf = Measure("dS", domain=mesh, subdomain_data=label_boundary_marker,
                      subdomain_id=intf_id)
    dx = Measure("dx", domain=mesh, subdomain_data=label_marker)
    if dom1_id > dom2_id:
        n = FacetNormal(mesh)("+")
    else:
        n = FacetNormal(mesh)("-")
    try:
        flow = np.array([assemble( dot(u, n)*ds_intf + Constant(0.0)*dx) for u in results["u"] ] )
    except:
        flow = np.array([assemble( dot(u("+"), n)*ds_intf + Constant(0.0)*dx) for u in results["u"] ] )
    return flow

if __name__=="__main__":
    mesh_name = sys.argv[1]
    sim_name = sys.argv[2]

    sim_config_file = f"results/{mesh_name}_{sim_name}/config.yml"
    mesh_config_file = f"meshes/{mesh_name}/{mesh_name}_config.yml"
    
    timeseries_file = f"results/{mesh_name}_{sim_name}/flow_timeseries.yml"

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
    source_expr, source_vals = get_source_expression(source_conf, mesh, subdomain_marker, porous_id, times)
    probes = mesh_config["probes"]
    flatprobes = dict(**probes["sas"],**probes["parenchyma"],**probes["ventricular_system"])
    domains = mesh_config["domains"]
    name_to_label = {dom["name"]:dom["id"] for dom in domains}

    V = VectorFunctionSpace(mesh, "CG", 2)
    infile = HDF5File(mesh.mpi_comm(), sim_file, "r")

    start_time = T - 1
    start_idx = np.where(times==start_time)[0][0] 

    timeseries = {}

    # extract source inflow data

    source_inflow = []
    for i in range(num_steps + 1):
        source_expr.i = i
        source_inflow.append(source_expr(Point([0,0,0])))
    source_inflow = np.array(source_inflow)
    dx_par = Measure("dx", domain=mesh, subdomain_data=label_marker, subdomain_id = name_to_label["parenchyma"])
    parenchyma_vol = assemble(Constant(1)*dx_par)
    inflow = source_inflow*parenchyma_vol

    timeseries["inflow"] = inflow

    results = {n:[] for n in names}
    times = times[start_idx:]

    for n in names:
        for i in range(num_steps + 1 - start_idx):
            f = Function(V)
            infile.read(f,f"{n}/vector_{i + start_idx}")
            results[n].append(f)
    infile.close()

    # compute outflow into spinal canal 
    ds_outflow = Measure("ds", domain=mesh, subdomain_data=boundary_marker, subdomain_id=spinal_outlet_id)
    n = FacetNormal(mesh)

    spinal_outflow = np.array([assemble(dot(u,n)*ds_outflow) for u in results["u"]])
    timeseries["spinal_outflow"] = spinal_outflow
    
    flow_pairs = [("lateral_ventricles", "foramina"),
                 ("aqueduct", "fourth_ventricle"),
                ("fourth_ventricle", "median_aperture"),
                ]

    for fp in flow_pairs:
        flow = compute_internal_flow(fp[0], fp[1])
        timeseries[f"{fp[0]} -> {fp[1]}"] = flow

    interf_id = 1 
    dS = Measure("dS", domain=mesh, subdomain_data=boundary_marker)
    dxD = Measure("dx", domain=mesh, subdomain_data=subdomain_marker)

    n = FacetNormal(mesh)("+")
    ds_Sig = dS(interf_id)

    interface_flow_over_time = [assemble(dot(u,n)*dS + Constant(0.0)*dxD) for u in results["u"]]

    timeseries["csf_interface_flow"] = interface_flow_over_time
    timeseries = {k: np.array(v).tolist() for k,v in timeseries.items()} 

    with open(timeseries_file, "w") as ts_file:
        yaml.dump(timeseries, ts_file)
