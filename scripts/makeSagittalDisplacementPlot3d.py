from brainpulse.PostProcessing import (load_results_and_mesh,
                                           scalar_bars,
                                           extract_data,
                                           scale_grid)
import matplotlib.pyplot as plt
import os
import numpy as np
import yaml
import sys
import pyvista as pv
pv.set_plot_theme("document")

name = "SagittalDisplacement"
fps = 5
interpFrames = 1
dist = 0.33
cpos = {"y":[(0, dist, 0), (0, 0, 0), (0, 0, 1)],
        "x":[(dist,0, 0), (0, 0, 0), (0, 0, 1)],
        "z": [(0, 0, dist*1.3), (0, 0, 0), (0, 1, 0)]}
origin = (0,0, 0.03)   
windows = [(1800,1600), (1400, 1600), (1600,1600)]   
views = ["x","y","z"]
invert_dict = {"x":True, "y":True, "z":True}

sargs = dict(title_font_size=80,label_font_size=80,shadow=True,n_labels=3,
             italic=True,font_family="arial", height=0.3, vertical=True)
scalar_bars = {"left": dict(position_x=0.05, position_y=0.02, **sargs),
               "right": dict(position_x=0.8, position_y=0.02, **sargs),
               "top_right":dict(position_x=0.8, position_y=0.65, **sargs),
               "top_left":dict(position_x=0.1, position_y=0.65, **sargs)}

class ImageGenerator(object):
    def __init__(self, mesh_name, sim_name, times=None):
        mesh_grid, sim_config, mesh_config, sim_file, source_expr =  load_results_and_mesh(mesh_name, sim_name)
        T = sim_config["T"]
        num_steps = sim_config["num_steps"]
        try:
            os.mkdir(plot3d_path)
        except FileExistsError:
            pass
        self.source_expr = source_expr
        self.times = np.linspace(0, T, num_steps + 1)
        if times is None:
            self.time_indices = list(range(num_steps + 1))
        else:
            self.time_indices =  []
            for pt in times:
                self.time_indices.append( np.abs(pt-self.times).argmin() ) 
  
        print("start loading data...")
        self.data_d = extract_data(mesh_grid, ["d"], ["parenchyma"],
                                    self.time_indices, sim_file, mesh_config)

        print("finished loading data.")

        self.data_d.clip((0,0,-1), (0,0,-0.1), inplace=True, invert=True)
        
        scale_grid(self.data_d, 1e3)

        self.d_range = self.get_range("d", self.data_d)
        

    def get_range(self, var, data):
        max_val = - np.inf
        min_val = + np.inf
        for time_idx in range(len(self.time_indices)):
            rng = data.get_data_range(f"{var}_{time_idx}")
            min_val = min([min_val, rng[0]])
            max_val = max([max_val, rng[1]])
        return [min_val, max_val]
        

    def generate_image(self, time_idx, view="x"):
        d = f"d_{int(time_idx)}"
        clipped_data_d = self.data_d.clip(view, origin=origin, invert=invert_dict[view])
        p = pv.Plotter(off_screen=True, notebook=False)
        show_scalar_bar = (view=="z")
        arrows_d = clipped_data_d.glyph(scale=d, factor=0.2, orient=d,absolute=False, tolerance=0.02)
        p.add_mesh(arrows_d, color ="white")
        p.add_mesh(clipped_data_d, scalars=d, cmap="amp", clim=(0,self.d_range[1]),
                   scalar_bar_args = scalar_bars["top_right"], stitle="disp [mm]",
                   show_scalar_bar=show_scalar_bar) 

        p.camera_position = cpos[view] #camera position, focal point, and view up.
        return p, ()


def create_array_plot(path, time_indices, source_expr, img_gen_func, times):

    nind = len(time_indices)
    size = 8
    print("start array plot generation")
    phases = ["I", "II", "III", "IV"]

    fig, axes = plt.subplots(nind, 3, figsize=(size*3.1, nind*size))
    for j, idx in enumerate(time_indices):
        for i,view in enumerate(views):
            p, _ = img_gen_func(j, view=view)
            img = p.screenshot(transparent_background=True, return_img=True,
                               window_size=windows[i],
                               filename=path + f"_{view}_{times[idx]:.3f}.png")
            p.clear()
            axes[j,i].imshow(img)
            if i==0:
                axes[j,i].set_title(f"Phase {phases[j]}",fontsize=30) 
            axes[j,i].axis('off')
    fig.tight_layout()
    fig.savefig(path + "_array_plot.pdf")

if __name__=="__main__":
    print("start plot3d creation...")
    mesh_name = sys.argv[1]
    sim_name = sys.argv[2] 
    plot3d_path = f"results/{mesh_name}_{sim_name}/plot3d/{name}"
    img_gen = ImageGenerator(mesh_name, sim_name)
    img_gen_func = lambda time_idx: img_gen.generate_image(time_idx)

    phase_times = [0.13, .35, 0.56, 0.8]
    img_gen = ImageGenerator(mesh_name, sim_name, times=phase_times)

    img_gen_func = lambda time_idx, view: img_gen.generate_image(time_idx, view=view)
    create_array_plot(f"{plot3d_path}/{name}", img_gen.time_indices, img_gen.source_expr, img_gen_func, img_gen.times)
