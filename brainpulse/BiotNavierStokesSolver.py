from dolfin import *
from multiphenics import *
import numpy as np
from pathlib import Path

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = "-O2 -ftree-vectorize"


def eps(u):
    return sym(grad(u))

def solve_biot_navier_stokes(mesh, T, num_steps,
                             material_parameter, 
                             boundary_marker,
                             subdomain_marker,
                             boundary_conditions,
                             porousrestriction,
                             fluidrestriction,
                             time_dep_expr=(),
                             elem_type="TH",
                             sliprate=0.0,
                             initial_pF=Constant(0.0),
                             initial_pP=Constant(0.0),
                             initial_phi=Constant(0.0),
                             initial_d=None,
                             initial_u=None,
                             g_source=Constant(0.0),
                             f_fluid=None,
                             f_porous=None,
                             outlet_pressure=Constant(0.0),
                             filename=None,
                             interf_id=1,
                             u_degree=2, p_degree=1):
    

    gdim = mesh.geometric_dimension()
    fluid_id = 2
    porous_id = 1
    spinal_outlet_id = 3

    for expr in time_dep_expr:
            expr.t = 0
            expr.i = 0

    # porous parameter
    c = material_parameter["c"]
    kappa = material_parameter["kappa"]
    lmbda = material_parameter["lmbda"]
    mu_s = material_parameter["mu_s"]
    rho_s = material_parameter["rho_s"]
    alpha = material_parameter["alpha"]

    # fluid parameter
    rho_f = material_parameter["rho_f"]
    mu_f = material_parameter["mu_f"]

    time = 0.0
    dt = Constant(T / num_steps)

    gamma = Constant(sliprate)
    gravity = Constant([0.0]*gdim)
    f = Constant([0.0]*gdim)

    n = FacetNormal(mesh)("+")
    

    dxF = Measure("dx", domain=mesh, subdomain_data=subdomain_marker, subdomain_id=fluid_id)
    dxP = Measure("dx", domain=mesh, subdomain_data=subdomain_marker, subdomain_id=porous_id)
    dxD = Measure("dx", domain=mesh, subdomain_data=subdomain_marker)
    dS = Measure("dS", domain=mesh, subdomain_data=boundary_marker)
    ds = Measure("ds", domain=mesh, subdomain_data=boundary_marker)

    ds_Sig = dS(interf_id)

    V = VectorFunctionSpace(mesh, "CG", u_degree)
    W = FunctionSpace(mesh, "CG", p_degree)
    H = BlockFunctionSpace([V, W, V, W, W],
                            restrict=[fluidrestriction, fluidrestriction,
                                    porousrestriction, porousrestriction,
                                    porousrestriction])
    

    trial = BlockTrialFunction(H)
    
    u, pF, d, pP, phi = block_split(trial)
    test = BlockTestFunction(H)
    v, qF, w, qP, psi = block_split(test)
    previous = BlockFunction(H)
    u_n, pF_n, d_n, pP_n, phi_n = block_split(previous)
    
    if initial_d is None:
        initial_d = Constant([0.0]*gdim)
    
    if initial_u is None:
        initial_u = Constant([0.0]*gdim)

    if f_fluid is None:
        f_fluid = Constant([0.0]*gdim)
    
    if f_porous is None:
        f_porous = Constant([0.0]*gdim)
    
    u_n.assign(interpolate(initial_u, H.sub(0)))
    pF_n.assign(interpolate(initial_pF, H.sub(1)))
    d_n.assign(interpolate(initial_d, H.sub(2)))
    pP_n.assign(interpolate(initial_pP, H.sub(3)))
    phi_n.assign(interpolate(initial_phi, H.sub(4)))

    previous.apply("from subfunctions")

    outlet_pressure_func = Function(W)

    # extract Dirichlet boundary conditions
    bcs = []
    for bc in boundary_conditions:
        for marker_id, subsp_bc_val in bc.items():
            for subspace_id, bc_val in subsp_bc_val.items():
                bc_d = DirichletBC(H.sub(subspace_id), bc_val,
                                    boundary_marker, marker_id)
                bcs.append(bc_d)

    bcs = BlockDirichletBC(bcs)

    def tang_interf(u,v):
        return dot(cross(u("+"),n), cross(v("+"),n))*ds_Sig + inner(u,v)*Constant(0.0)*dxD

    def a_F(u,v):
            return rho_f*dot(u/dt, v)*dxF \
                    + 2*mu_f*inner(eps(u), eps(v))*dxF \
                    + (gamma*mu_f/sqrt(kappa))*tang_interf(u,v)

    # define forms

    def b_1_F(v, qF):
        return  -qF*div(v)*dxF

    def b_2_Sig(v, qP):
        return qP("+")*inner(v("+"), n)*ds_Sig + div(v)*qP*Constant(0.0)*dxD

    def b_3_Sig(v, d):
        return - ((gamma*mu_f/sqrt(kappa))*tang_interf(v, d))

    def b_4_Sig(w,qP):
        return -qP("+") * dot(w("+"),n)*ds_Sig + div(w)*qP*Constant(0.0)*dxD

    def a_1_P(d, w):
        return 2.0*mu_s*inner(eps(d), eps(w))*dxP \
                + (gamma*mu_f/sqrt(kappa))*tang_interf(d/dt,w)

    def b_1_P(w, psi):
        return - psi*div(w)*dxP

    def a_2_P(pP,qP):
        return (kappa/mu_f) *inner(grad(pP), grad(qP))*dxP \
                + (c + alpha**2/lmbda)*(pP/dt)*qP*dxP

    def b_2_P(psi, qP):
        return (alpha/lmbda)*psi*qP*dxP

    def a_3_P(phi, psi):
        return (1.0/lmbda)*phi*psi*dxP

    def F_F(v):
        return rho_f*dot(f_fluid, v)*dxF \
             + dot(outlet_pressure_func*FacetNormal(mesh), v)*ds(spinal_outlet_id)

    def F_P(w):
        return rho_s*inner(f_porous, w)*dxP

    def G(qP):
        return rho_f*inner(gravity, grad(qP))*dxP \
                - rho_f*inner(gravity, n)*qP("+")*ds_Sig  + qP*Constant(0.0)*dxD \
                + g_source*qP*dxP

    def F_F_n(v):
        return F_F(v) + rho_f*inner(u_n/dt, v)*dxF + b_3_Sig(v, d_n/dt)

    def F_P_n(w):
        return F_P(w) + (gamma*mu_f/sqrt(kappa))*tang_interf(d_n/dt, w)

    def G_n(qP):
        return G(qP) + (c + (alpha**2)/lmbda)*pP_n/dt*qP*dxP \
            - b_4_Sig(d_n/dt, qP) - b_2_P(phi_n/dt, qP)


    # define system:
    # order trial: u, pF, d, pP, phi
    # order test: v, qF, w, qP, psi

    lhs = [[a_F(u,v)         , b_1_F(v, pF), b_3_Sig(v, d/dt) , b_2_Sig(v, pP), 0                ],
        [ b_1_F(u, qF)       ,  0          , 0                , 0             , 0                ],
        [ b_3_Sig(u, w)      ,  0          , a_1_P(d,w)       , b_4_Sig(w, pP), b_1_P(w, phi)    ],
        [ b_2_Sig(u, qP)     ,  0          , b_4_Sig(d/dt, qP), -a_2_P(pP, qP), b_2_P(phi/dt, qP)], 
        [ 0                  ,  0          , b_1_P(d, psi)    , b_2_P(psi, pP), -a_3_P(phi, psi) ]]

    AA = block_assemble(lhs, keep_diagonal=True)
    bcs.apply(AA)
    solver = PETScLUSolver(AA, "mumps")

    if filename:
        hdf5_file = HDF5File(mesh.mpi_comm(), f"{Path(filename).parent}/{Path(filename).stem}.hdf", "w")
        output_legacy = XDMFFile(f"{Path(filename).parent}/{Path(filename).stem}.xdmf" )
        output_legacy.parameters["functions_share_mesh"] = True
        output_legacy.parameters["rewrite_function_mesh"] = False

    names = ["u", "pF", "d", "pP", "phi"]

    def solve():
        rhs = [F_F_n(v), 0, F_P_n(w), -G_n(qP), 0]
        FF = block_assemble(rhs)
        bcs.apply(FF)
        solver.solve(previous.block_vector(), FF)
        previous.block_vector().block_function().apply("to subfunctions")
     
    results = block_split(previous)
    write_to_file(results, time, names, hdf5_file, output_legacy, elem_type)
    tot_outflow = 0.0
    outflow = 0.0
    outlet_pressure_func.assign(interpolate(outlet_pressure, W))
    for i in range(num_steps):
        time = (i + 1)*dt.values()[0]
        for expr in time_dep_expr:
            expr.t = time
            expr.i = i + 1
            expr.outflow_vol = float(tot_outflow)
            expr.outflow = float(outflow)
            expr.p = results[1]
        outlet_pressure_func.assign(interpolate(outlet_pressure, W))
                
        solve()
        results = block_split(previous)

        write_to_file(results, time, names, hdf5_file,
                      output_legacy, elem_type)
        
        u = results[0]
        outflow = assemble(inner(u,FacetNormal(mesh))*ds(spinal_outlet_id))
        tot_outflow += float(outflow)*dt.values()[0]

    hdf5_file.close()
    output_legacy.close()
    return results


def write_to_file(results, time, names, hdf5_file, output_legacy, elem_type):
    for k,r in enumerate(results):
        r.rename(names[k], names[k])
        output_legacy.write(r, time)
        hdf5_file.write(r, r.name(), time)
