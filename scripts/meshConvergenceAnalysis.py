import yaml
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain

plt.style.use("bmh")
plt.rcParams['axes.facecolor'] = 'white' 
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['xtick.labelsize'] = 16 
plt.rcParams['ytick.labelsize'] = 16 
labelsize = 15

m3tomL = 1e6
figsize = (4.5, 3)

models = {  "coarse":"coarseBrainMesh_standardNt960",
            "mid":"midBrainMesh_standardNt960",
            "fine":"fineBrainMesh_standardNt960",}

time_conv_models = {"$N_T=240$" : "fineBrainMesh_standardNt240",
                    "$N_T=480$" : "fineBrainMesh_standardNt480",
                    "$N_T=960$" : "fineBrainMesh_standardNt960",
                    }


key_results = {name:{} for name in {**models, **time_conv_models}.keys()}

key_quantities = [("Peak Transmantle Pressure Gradient","peak_pressure_gradient"),
                  ("LV Pressure Amplitude","pressure_amp"),
                  ("Spinal Stroke Volume","spinal_out_vol"),
                  ("Aquaeduct Stroke Volume","ASV"),
                  ("Peak Aqueduct Flow Rate","PVF"),
                  ("Peak Displacement","max_displacement"),
                  ("peak ISF velocity","max_parenchymal_flow")
                ]

for k,v in chain(models.items(), time_conv_models.items()):
    with open(f"results/{v}/pressure_key_quantities.yml", 'r') as ff:
        results = yaml.full_load(ff)
        key_results[k].update(results)
    with open(f"results/{v}/flow_key_quantities.yml", 'r') as fp:
        results = yaml.full_load(fp)
        key_results[k].update(results)
    with open(f"results/{v}/displacement_key_quantities.yml", 'r') as fp:
        results = yaml.full_load(fp)
        key_results[k].update(results)
    key_results[k]["spinal_out_vol"] = (key_results[k]["max_cum_spinal_outflow"]
     - key_results[k]["min_cum_spinal_outflow"])*m3tomL
    key_results[k]["pressure_amp"] = (key_results[k]["max_lat_ventricle_pressure"]
     - key_results[k]["min_lat_ventricle_pressure"])
    try:
        key_results[k]["PVF"] = (key_results[k]["max_flow_aqueduct -> fourth_ventricle"]
            - key_results[k]["min_cum_flow_aqueduct -> fourth_ventricle"])*m3tomL
        key_results[k]["ASV"] = (key_results[k]["max_cum_flow_aqueduct -> fourth_ventricle"]
     - key_results[k]["min_cum_flow_aqueduct -> fourth_ventricle"])*m3tomL
    except KeyError:
        key_results[k]["PVF"] = (key_results[k]["max_flow_third_ventricle -> fourth_ventricle"]
            - key_results[k]["min_flow_third_ventricle -> fourth_ventricle"])*m3tomL
        key_results[k]["ASV"] = (key_results[k]["max_cum_flow_third_ventricle -> fourth_ventricle"]
     - key_results[k]["min_cum_flow_third_ventricle -> fourth_ventricle"])*m3tomL
    key_results[k]["max_displacement"] *= 1e3
    key_results[k]["hmax"] *= 1e3
    key_results[k]["hmin"] *= 1e3
    key_results[k]["max_parenchymal_flow"]*1e6



units = {"Peak Transmantle Pressure Gradient":"mmHg/m",
          "LV Pressure Amplitude":"mmHg",
          "Spinal Stroke Volume":"ml",
          "Aquaeduct Stroke Volume":"ml",
          "Peak Aqueduct Flow Rate":"ml/s",
          "Peak Displacement":"mm",
          "Number of cells":None,
          "Number of vertices":None,
          "min cell diameter":"mm",
          "max cell diameter":"mm",
          "peak ISF velocity":"$\mu m/s"
         }

for kq in [("Number of cells","num_cells"),
           ("Number of vertices","num_vertices"),
           ("max cell diameter","hmax"),
           ("min cell diameter","hmin")]:
    m = list(models.keys())
    plt.figure(figsize=figsize)
    plt.bar(m, [key_results[model][kq[1]] for model in m])
    plt.title(kq[0],fontsize=labelsize)
    plt.ylabel(units[kq[0]], fontsize=labelsize)
    plt.box(False)
    plt.tight_layout()
    plt.savefig(f"figures/bar_conv_{kq[1]}.pdf")
    plt.close()

for kq in key_quantities:
    m = list(models.keys())
    plt.figure(figsize=figsize)
    x = [key_results[model]["num_cells"] for model in m]
    y = [key_results[model][kq[1]] for model in m]
    plt.plot(x,y, "-*")
    plt.ylim((0, (np.array(y).max()*1.1)))
    plt.ylabel(units[kq[0]], fontsize=labelsize)
    plt.xlabel("# elements", fontsize=labelsize)
    plt.title(kq[0],fontsize=labelsize)
    plt.tight_layout()
    plt.savefig(f"figures/conv_{kq[1]}.pdf")
    plt.close()


for kq in key_quantities:
    m = list(time_conv_models.keys())
    plt.figure(figsize=figsize)
    x = [80, 160, 320]
    y = [key_results[model][kq[1]] for model in m]
    plt.plot(x,y, "-*")
    plt.ylim((0, (np.array(y).max()*1.1)))
    plt.ylabel(units[kq[0]], fontsize=labelsize)
    plt.xlabel("# timesteps", fontsize=labelsize)
    plt.title(kq[0],fontsize=labelsize)
    plt.tight_layout()
    plt.savefig(f"figures/time_conv_{kq[1]}.pdf")
    plt.close()

