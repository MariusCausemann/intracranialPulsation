import yaml
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("bmh")

m3tomL = 1e6
models = {"Standard":"fineBrainMesh_standardNt960",
          "Model A" :"fineBrainMesh_ModelA",
          "Model B" :"fineBrainMesh_ModelB",
          "Model C" :"fineBrainMesh_ModelC",
          "Model D" :"fineBrainMesh_ModelD",
        }


model_deviation = {"Standard":" - ",
                    "Model A" :r"PVI =10$\,$ml",
                    "Model B" :r"E=3000$\,$Pa",
                    "Model C" :r" $\nu$=0.4",
                    "Model D" :r"c=1e-5$\,$Pa$^{-1}$",
                    }

key_results = {name:{} for name in models.keys()}

key_quantities = [("Peak Transmantle Pressure Gradient [mmHg/m]","peak_pressure_gradient"),
                  ("LV Pressure Amplitude [mmHg]","pressure_amp"),
                  ("Spinal Stroke Volume [ml]","spinal_out_vol"),
                  ("Aqueduct Stroke Volume [ml]","ASV"),
                  ("Peak Aqueduct Flow Rate [ml/s]","PVF"),
                  ("Peak Displacement [mm]","max_displacement"),
                ]

for k,v in models.items():
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

bounds = {"Peak Transmantle Pressure Gradient [mmHg/m]":(1.46-0.74,1.46+0.74),
          "LV Pressure Amplitude [mmHg]":(5,10),
          "Spinal Stroke Volume [ml]":(1,2),
          "Aqueduct Stroke Volume [ml]":(0.03, 0.05),
          "Peak Aqueduct Flow Rate [ml/s]":(0.31-0.16,0.31+0.16),
          "Peak Displacement [mm]":(0.1,0.5)
         }

units = {"Peak Transmantle Pressure Gradient [mmHg/m]":"mmHg/m",
          "LV Pressure Amplitude [mmHg]":"mmHg",
          "Spinal Stroke Volume [ml]":"ml",
          "Aqueduct Stroke Volume [ml]":"ml",
          "Peak Aqueduct Flow Rate [ml/s]":"ml/s",
         "Peak Displacement [mm]":"mm"
         }

#plt.rcParams.update({'font.size': 14})
labelsize = 20
plt.rcParams['xtick.labelsize'] = labelsize 
plt.rcParams['ytick.labelsize'] = labelsize 

for kq in key_quantities:
    m = list(models.keys())

    plt.figure(figsize=(1.6*len(m), 4))
    x = np.arange(-0.5, len(m))
    plt.fill_between(x, bounds[kq[0]][0], bounds[kq[0]][1], alpha=0.2)
    plt.bar([model + f" \n {model_deviation[model]}"  for model in m], [key_results[model][kq[1]] for model in m])
    std_value = key_results["Standard"][kq[1]]
    for i,model in enumerate(m):
        val = key_results[model][kq[1]]
        plt.annotate(f"{100*val/std_value:.0f}%", (i- 0.2, val + bounds[kq[0]][1]*0.02), fontsize=labelsize - 2)
    plt.title(kq[0], fontsize=labelsize)
    plt.hlines((bounds[kq[0]][0] + bounds[kq[0]][1])/2, x[0], x[-1] )
    plt.ylim(0, bounds[kq[0]][1])
    plt.ylabel(units[kq[0]], fontsize=labelsize)
    plt.box(False)
    plt.xticks(rotation=-40)
    plt.savefig(f"figures/bar_{kq[1]}.pdf", bbox_inches = "tight")