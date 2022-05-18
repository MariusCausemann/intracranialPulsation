
import yaml

real_brain_simulations = [("coarseBrainMesh","standardNt240"),
                          #("coarseBrainMesh","standardNt960"),
                          #("midBrainMesh","standardNt960"),
                          #("fineBrainMesh","standardNt960"),
                          #("fineBrainMesh","standardNt480"),
                          #("fineBrainMesh","standardNt240"),
                          #("fineBrainMesh","ModelA"),
                          #("fineBrainMesh","ModelB"),
                          #("fineBrainMesh","ModelC"),
                          #("fineBrainMesh","ModelD"),
                        ]

sing_image = "biotstokes_openblas.simg"
sing_image = "docker://mcause/brainsim:openblas"
mpi_command = "srun"
meshes = ["coarseBrainMesh", "midBrainMesh", "fineBrainMesh"]

plot3d = ["PressureFlow", "SagittalPressure", "SagittalDisplacement"] 

def estimate_resources(wildcards, input, attempt):
    with open(input.config,"r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    with open("config_files/computational_resources.yml","r") as config_file:
        resources = yaml.load(config_file, Loader=yaml.FullLoader)

    resources = resources[wildcards.mesh]
    cpus = resources["cpus"]
    mem = resources["mem"]
    total_secs = resources["secs_for_factorization"] + resources["secs_per_solve"]*config["num_steps"]

    tot_minutes = total_secs//60
    mins = tot_minutes%60
    hours = tot_minutes//60
    
    return {"mem_mb":mem, "cpus":cpus, "nodes":1, "time":f"0{hours}:{mins}:00"}


rule all:
    input:
        expand("results/{sim}/plot3d/{plot3d}/{plot3d}_array_plot.pdf", plot3d=plot3d,
                sim=[f"{mesh}_{sim_name}" for mesh, sim_name in real_brain_simulations ]),
        expand("results/{sim}/flow_key_quantities.yml",
                    sim=[f"{mesh}_{sim_name}" for mesh, sim_name in real_brain_simulations ]),
        expand("results/{sim}/pressure_key_quantities.yml",
                    sim=[f"{mesh}_{sim_name}" for mesh, sim_name in real_brain_simulations ]),
        expand("results/{sim}/displacement_key_quantities.yml",
                    sim=[f"{mesh}_{sim_name}" for mesh, sim_name in real_brain_simulations ]),

rule extractMeshes:
    input:
        "meshes/meshes.tar.gz.aa",
        "meshes/meshes.tar.gz.ab",
        "meshes/meshes.tar.gz.ac",
        "meshes/meshes.tar.gz.ad",
    output:
        expand("meshes/{mesh}/{mesh}.h5", mesh=meshes),
        expand("meshes/{mesh}/{mesh}.xdmf", mesh=meshes),
        expand("meshes/{mesh}/{mesh}_boundaries.h5", mesh=meshes),
        expand("meshes/{mesh}/{mesh}_boundaries.xdmf", mesh=meshes),
    shell:
        """
        singularity exec \
        {sing_image} bash -c \
        "cd meshes && cat meshes.tar.gz.* | tar xzv --strip-components=1"
        """



rule runBrainSim:
    input:
        "meshes/{mesh}/{mesh}.h5",
        "meshes/{mesh}/{mesh}_boundaries.xdmf",
        "meshes/{mesh}/{mesh}_boundaries.h5",
        config = "config_files/{sim_name}.yml",
        meshfile = "meshes/{mesh}/{mesh}.xdmf",
    output:
        "results/{mesh}_{sim_name}/results.hdf",
        outfile="results/{mesh}_{sim_name}/results.xdmf",
        config="results/{mesh}_{sim_name}/config.yml",
    log:
        "results/{mesh}_{sim_name}/log"
    resources:
        ntasks=lambda wildcards, input, attempt: estimate_resources(wildcards, input, attempt)["cpus"],
        time=lambda wildcards, input, attempt: estimate_resources(wildcards, input, attempt)["time"],
    shell:
        """
        {mpi_command} -n {resources.ntasks} \
        singularity exec \
        {sing_image} \
        python3 scripts/runFluidPorousBrainSim.py \
        -c {input.config} \
        -m {input.meshfile} \
        -o {output.outfile} && \
        cp {input.config} {output.config}
        """

rule makePressurePlots:
    input:
        sim_config_file="results/{mesh}_{sim_name}/config.yml",
        config = "config_files/{sim_name}.yml",
        data="results/{mesh}_{sim_name}/pressure_timeseries.yml",
    output:
        "results/{mesh}_{sim_name}/pressure_key_quantities.yml"
    shell:
         """
        singularity exec \
        {sing_image} \
        xvfb-run -a python3 scripts/pressurePlots.py \
        {wildcards.mesh} {wildcards.sim_name}
        """

rule extractPressureData:
    input:
        sim_results="results/{mesh}_{sim_name}/results.hdf",
        subdomain_file="meshes/{mesh}/{mesh}.xdmf",
        sim_config_file="results/{mesh}_{sim_name}/config.yml",
        config = "config_files/{sim_name}.yml",
    output:
        "results/{mesh}_{sim_name}/pressure_timeseries.yml"
    resources:
        ntasks=lambda wildcards, input, attempt: estimate_resources(wildcards, input, attempt)["cpus"]/2, 
    shell:
         """
        {mpi_command} -n {resources.ntasks} \
        singularity exec \
        {sing_image} \
        xvfb-run -a python3 scripts/extractPressureData.py \
        {wildcards.mesh} {wildcards.sim_name}
        """

rule makeFlowPlots:
    input:
        sim_config_file="results/{mesh}_{sim_name}/config.yml",
        config = "config_files/{sim_name}.yml",
        data = "results/{mesh}_{sim_name}/flow_timeseries.yml",
    output:
        "results/{mesh}_{sim_name}/flow_key_quantities.yml"
    resources:
        ntasks=lambda wildcards, input, attempt: estimate_resources(wildcards, input, attempt)["cpus"]/2, 
    shell:
         """
        singularity exec \
        {sing_image} \
        xvfb-run -a python3 scripts/flowPlots.py \
        {wildcards.mesh} {wildcards.sim_name}
        """


rule extractFlowData:
    input:
        sim_results="results/{mesh}_{sim_name}/results.hdf",
        subdomain_file="meshes/{mesh}/{mesh}.xdmf",
        sim_config_file="results/{mesh}_{sim_name}/config.yml",
        config = "config_files/{sim_name}.yml",
    output:
        "results/{mesh}_{sim_name}/flow_timeseries.yml"
    resources:
        ntasks=lambda wildcards, input, attempt: estimate_resources(wildcards, input, attempt)["cpus"]/2, 
    shell:
         """
        {mpi_command} -n {resources.ntasks} \
        singularity exec \
        {sing_image} \
        xvfb-run -a python3 scripts/extractFlowData.py \
        {wildcards.mesh} {wildcards.sim_name}
        """

rule makeDisplacementPlots:
    input:
        sim_results="results/{mesh}_{sim_name}/results.hdf",
        subdomain_file="meshes/{mesh}/{mesh}.xdmf",
        sim_config_file="results/{mesh}_{sim_name}/config.yml",
        config = "config_files/{sim_name}.yml",
    output:
        "results/{mesh}_{sim_name}/displacement_key_quantities.yml"
    resources:
        ntasks=lambda wildcards, input, attempt: estimate_resources(wildcards, input, attempt)["cpus"], 
    shell:
         """
        singularity exec \
        {sing_image} \
        xvfb-run -a python3 scripts/displacementPlots.py \
        {wildcards.mesh} {wildcards.sim_name}
        """


rule makeplot3d:
    input:
        sim_results="results/{mesh}_{sim_name}/results.xdmf",
        subdomain_file="meshes/{mesh}/{mesh}.xdmf",
        sim_config_file="results/{mesh}_{sim_name}/config.yml",
    output:
        "results/{mesh}_{sim_name}/plot3d/{plot3d_name}/{plot3d_name}_array_plot.pdf",
    shell:
        """
        singularity exec \
        {sing_image} \
        xvfb-run -a python3 scripts/make{wildcards.plot3d_name}Plot3d.py \
        {wildcards.mesh} {wildcards.sim_name}
        """

