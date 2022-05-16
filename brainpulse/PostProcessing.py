import matplotlib.pyplot as plt
from fenics import *
from multiphenics import *
import numpy as np
from brainpulse.IOHandling import xdmf_to_unstructuredGrid
from brainpulse.SourceExpression import get_source_expression
from matplotlib.backends.backend_agg import FigureCanvasAgg
import yaml
import pyvista as pv
import imageio

ventricular_system = ["lateral_ventricles", "foramina", "aqueduct", "third_ventricle", "fourth_ventricle",
                      "median_aperture"]

sargs = dict(title_font_size=20,label_font_size=16,shadow=True,n_labels=3,
             italic=True,font_family="arial", height=0.4, vertical=True, position_y=0.05)
scalar_bars = {"left": dict(position_x=0.1, **sargs),
               "right": dict(position_x=0.9, **sargs)}

porous_id = 1

def project_list(v_list, V):
    w = TestFunction(V)
    Pv = TrialFunction(V)
    mesh = V.mesh()
    dx = Measure("dx",mesh)
    a = inner(w, Pv) * dx

    # Assemble linear system
    A = assemble(a)

    solver = PETScLUSolver(as_backend_type(A), "mumps")

    # Solve linear system for projection
    results = []
    for v in v_list:
        f = Function(V)
        rhs = assemble(inner(w, v) * dx)
        solver.solve(f.vector(), rhs)
        results.append(f)

    return results

def scale_grid(grid, fac):
    for name, pdata in grid.point_arrays.items():
        grid.point_arrays[name] *= fac
        
def sum_grids(grid1, grid2):
    for name, pdata in grid1.point_arrays.items():
        grid1.point_arrays[name] += grid2.point_arrays[name]
    return grid1


def compute_stat(stat, mg, var, parts, idx, sim_file, mesh_config):
    data = extract_data(mg, var, parts, int(idx), sim_file, mesh_config)
    return stat(data[var])

def compute_glob_stat(stat, mg, var, parts, indices, sim_file, mesh_config):
    return stat([compute_stat(stat, mg, var, parts, idx, sim_file, mesh_config) for idx in indices])

def extract_and_interpolate(mg, var, parts, idx, sim_file, mesh_config):
    print(idx)
    if np.isclose(idx%1, 0.0):
        return extract_data(mg, var, parts, int(idx),sim_file, mesh_config)
    
    print(f"start interpolating for idx {idx}")
    floor =  int(np.floor(idx))
    ceil =  int(np.ceil(idx))
    data1 = extract_data(mg, var, parts, floor, sim_file, mesh_config)
    data2 = extract_data(mg, var, parts, ceil, sim_file, mesh_config)
    
    scale_grid(data1, idx - floor)
    scale_grid(data2, ceil - idx)
    data1 = sum_grids(data1, data2)
    return data1

def extract_data(mg, var, parts, idx, sim_file, mesh_config):
    mg = mg.copy()
    if isinstance(var, str):
        var = [var]
    if isinstance(idx, int):
        idx = [idx]
    grid = xdmf_to_unstructuredGrid(sim_file, variables=var, idx=idx)
     # add new data to mesh
    for name, data in grid.point_arrays.items():
        mg.point_arrays[name] = grid.point_arrays[name]
    #filter parts:
    dom_meshes= []
    for dom in mesh_config["domains"]:
        if dom["name"] in parts:
            dom_meshes.append(mg.threshold([dom["id"],dom["id"]],
                                            scalars="subdomains"))
    merged = dom_meshes[0].merge(dom_meshes[1:])
    return merged


def plot_partial_3D(mg, idx, scenes, sim_file, mesh_config, cpos, interactive=False):
    if interactive:
        p = pv.PlotterITK()
    else:
        p = pv.Plotter(off_screen=True, notebook=False)
    max_val = -np.inf
    min_val = np.inf
    data_dict = {}
    for scene in scenes:
        var = scene["var"]
        parts = scene["mesh_parts"]
        data = extract_and_interpolate(mg, var, parts, idx, sim_file, mesh_config)
        if "clip" in scene.keys():
            try:
                data = data.clip(*scene["clip"])
            except:
                data = data.clip(**scene["clip"])
        if "slice" in scene.keys():
            try:
                data = data.slice(*scene["slice"])
            except:
                data = data.slice(**scene["slice"])
        data_dict[var] = data
        if "arrow" in scene.keys():
            continue
        max_val = max(max_val, data[var].max())
        min_val = min(min_val, data[var].min())
    
    for i, scene in enumerate(scenes):
        var = scene["var"]
        parts = scene["mesh_parts"]
        data =  data_dict[var]
        if "warp" in scene.keys():
            data = data.warp_by_vector(var, scene["warp_fac"])
        if interactive:
            options = scene["interactive"]
        else:
            options = scene["static"]
            if "clim" not in options.keys() or options["clim"] is None:
                pass
                #options["clim"] = (min_val, max_val)
        if "arrow" in scene.keys():
            vec_scale = scene["vec_scale"]
            arrows = data.glyph(scale=var, factor=vec_scale, orient=var)
            p.add_mesh(arrows,**options)#, lighting=False) #stitle=f"{var} Magnitude",
        else:
            p.add_mesh(data, scalars=var,**options)
    #camera position, focal point, and view up.
    p.camera_position = cpos
    return p, (min_val, max_val)


def plot_source_rgb_raw(source_series, times, t, size, dpi):
    fig = plt.Figure(figsize=size, dpi=dpi)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting here
    ax = fig.add_subplot(111)
    ax.plot(times, source_series, "-*")
    ax.axvline(t, color="red")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("g in 1/s")
    ax.grid()
    fig.tight_layout()
    # Retrieve a view on the renderer buffer
    canvas.draw()
    buf = canvas.buffer_rgba()
    # convert to a NumPy array
    data = np.asarray(buf)
    return data[:,:,:]


# create video
def create_movie(path, times, source_expr, plot_generator, fps, interpolate_frames):
    source_series = []
    for i in range(len(times)):
        #source_expr.t=t
        source_expr.i=i
        source_series.append(source_expr(Point([0,0,0])))
    try:
        rep_idx = np.argwhere(np.array(times) > 1/source_expr.f)[0][0]
    except:
        rep_idx = len(times) - 1
    rep_idx = len(times) - 1
    dt = times[1]
    print(source_series)
    mwriter = imageio.get_writer(f"{path}.mp4", fps=fps)
    for i,t in enumerate(times):
        print(f"rendering frame for time {t}...")
        for k in range(interpolate_frames):
            p,_= plot_generator(i + k/interpolate_frames)
            p.show(interactive=False, auto_close=False)
            img = p.screenshot(transparent_background=True, return_img=True, window_size=None)
            #p.deep_clean()
            p.close()
            size = (4,3)
            dpi = 70
            miniplot = plot_source_rgb_raw(source_series[:rep_idx], times[:rep_idx],
                                           (t + k/interpolate_frames*dt)%times[rep_idx] , 
                                           size, dpi)
            x,y,z = miniplot.shape
            img[:x,:y,:] = miniplot
            mwriter.append_data(img)
            imageio.imwrite(f"{path}_{i:03d}.png", img)
            if i==len(times) - 1:
                break

    mwriter.close()

def load_results_and_mesh(mesh_name, sim_name):

    sim_file_old = f"results/{mesh_name}_{sim_name}/results.xdmf"
    sim_file = f"results/{mesh_name}_{sim_name}/results.xdmf"

    movie_path = f"results/{mesh_name}_{sim_name}/movies/"
    sim_config_file = f"results/{mesh_name}_{sim_name}/config.yml"
    mesh_config_file = f"meshes/{mesh_name}/{mesh_name}_config.yml"
    label_file = f"meshes/{mesh_name}/{mesh_name}_labels.xdmf"

    with open(mesh_config_file) as conf_file:
        mesh_config = yaml.load(conf_file, Loader=yaml.FullLoader)

    with open(sim_config_file) as conf_file:
        sim_config = yaml.load(conf_file, Loader=yaml.FullLoader)
        
    mesh_grid = pv.read(label_file)

    T = sim_config["T"]
    num_steps = sim_config["num_steps"]
    dt = T/num_steps
    times = np.linspace(0, T, num_steps + 1)
    source_conf = sim_config["source_data"]
    mesh, subdomain_marker, label_marker, boundary_marker, label_boundary_marker = load_meshes(mesh_name)

    source_expr,_ = get_source_expression(source_conf, mesh, subdomain_marker, porous_id, times)
    
    return mesh_grid, sim_config, mesh_config, sim_file_old, source_expr



def load_meshes(mesh_name):
    mesh_dir = f"meshes/{mesh_name}"
    mesh_config_file = f"{mesh_dir}/{mesh_name}_config.yml"
    mesh_file = f"{mesh_dir}/{mesh_name}.xdmf"

    boundary_file = f"{mesh_dir}/{mesh_name}_boundaries.xdmf"
    label_boundary_file = f"{mesh_dir}/{mesh_name}_label_boundaries.xdmf"
    label_file = f"{mesh_dir}/{mesh_name}_labels.xdmf"

    infile_mesh = XDMFFile(mesh_file)
    mesh = Mesh()
    infile_mesh.read(mesh)
    gdim = mesh.geometric_dimension()
    subdomain_marker = MeshFunction("size_t", mesh, gdim)
    infile_mesh.read(subdomain_marker)#, "subdomains"
    infile_mesh.close()

    label_marker = MeshFunction("size_t", mesh, gdim, 0)
    label_marker_infile = XDMFFile(label_file)
    label_marker_infile.read(label_marker)
    label_marker_infile.close()

    boundary_marker = MeshFunction("size_t", mesh, gdim - 1, 0)
    boundary_infile = XDMFFile(boundary_file)
    boundary_infile.read(boundary_marker)
    boundary_infile.close()

    label_boundary_marker = MeshFunction("size_t", mesh, gdim - 1, 0)
    label_boundary_infile = XDMFFile(label_boundary_file)
    label_boundary_infile.read(label_boundary_marker)
    label_boundary_infile.close()

    return mesh, subdomain_marker, label_marker, boundary_marker, label_boundary_marker

