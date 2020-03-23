# import _thread
import time
import threading
from casadi import *
from matplotlib import pyplot as plt
import numpy as np

# add classes
from Define_parameters import Parameters
from Define_casadi_callback import AnimateCallback

# add fcn
from LoadData import load_data_markers, load_data_emg, load_data_GRF
from Fcn_InitialGuess import load_initialguess_muscularExcitation, load_initialguess_q
from Marche_Fcn_Integration import int_RK4_swing, int_RK4_stance, int_RK4
from Fcn_forward_dynamic import ffcn_contact, ffcn_no_contact
from Fcn_Objective import fcn_objective_activation, fcn_objective_emg, fcn_objective_markers, fcn_objective_GRF
from Fcn_print_data import save_GRF_real, save_Markers_real, save_EMG_real, save_params, save_bounds, save_initialguess

# SET PARAMETERS
params = Parameters()

# ----------------------------- States & controls ----------------------------------------------------------------------
# CONTROL
u  = MX.sym("u", params.nbU)
a  = u[:params.nbMus]                                         # muscular activation
F  = u[params.nbMus:]                                         # articular torque

# PARAMETERS
p  = MX.sym("p", params.nP)                                   # maximal isometric force adjustment

# STATE
x  = MX.sym("x", params.nbX)
q  = x[:params.nbQ]                                           # generalized coordinates
dq = x[params.nbQ: 2 * params.nbQ]                            # velocities


# ----------------------------- Load Data from c3d file ----------------------------------------------------------------
# GROUND REACTION FORCES & SET TIME
[GRF_real, params.T, params.T_stance] = load_data_GRF(params, 'cycle')
params.T_swing                        = params.T - params.T_stance

# MARKERS POSITION
M_real_stance = load_data_markers(params, 'stance')
M_real_swing  = load_data_markers(params, 'swing')

M_real          = np.zeros((3, params.nbMarker, (params.nbNoeuds + 1)))
M_real[0, :, :] = np.hstack([M_real_stance[0, :, :], M_real_swing[0, :, :]])
M_real[1, :, :] = np.hstack([M_real_stance[1, :, :], M_real_swing[1, :, :]])
M_real[2, :, :] = np.hstack([M_real_stance[2, :, :], M_real_swing[2, :, :]])

# MUSCULAR EXCITATION
U_real_swing  = load_data_emg(params, 'swing')
U_real_stance = load_data_emg(params, 'stance')
U_real        = np.hstack([U_real_stance, U_real_swing])


# ----------------------------- Movement -------------------------------------------------------------------------------
U = MX.sym("U", params.nbU * params.nbNoeuds)          # controls
X = MX.sym("X", params.nbX * (params.nbNoeuds + 1))    # states
P = MX.sym("P", params.nP)                             # parameters
G = []                                                 # equality constraints
Ja = 0                                                 # objective function for muscle activation
Jm = 0                                                 # objective function for markers
Je = 0                                                 # objective function for EMG
JR = 0                                                 # objective function for ground reactions

# ------------ PHASE 1 : Stance phase
for k in range(params.nbNoeuds_stance):
    # DYNAMIQUE
    Uk = U[params.nbU*k: params.nbU*(k + 1)]
    Xk = X[params.nbX*k: params.nbX*(k + 1)]
    G.append(X[params.nbX * (k + 1): params.nbX * (k + 2)] - int_RK4(ffcn_contact, params, Xk, Uk, P))
    # G.append(X[params.nbX*(k + 1): params.nbX*(k + 2)] - int_RK4_stance(params.T_stance, params.nbNoeuds_stance, params.nkutta,  Xk, Uk))

    # OBJECTIVE FUNCTION
    [grf, Jr] = fcn_objective_GRF(params.wR, Xk, Uk, GRF_real[:, k])                                                    # tracking ground reaction --> stance
    JR += Jr
    Jm += fcn_objective_markers(params.wMa, params.wMt, Xk[: params.nbQ], M_real_stance[:, :, k], 'stance')             # tracking marker
    Je += fcn_objective_emg(params.wU, Uk, U_real_stance[:, k])                                                         # tracking emg
    Ja += fcn_objective_activation(params.wL, Uk)                                                                       # min muscle activations (no EMG)

# ------------ PHASE 2 : Swing phase
for k in range(params.nbNoeuds_swing):
    # DYNAMIQUE
    Uk = U[params.nbU * params.nbNoeuds_stance + params.nbU*k: params.nbU * params.nbNoeuds_stance + params.nbU*(k + 1)]
    Xk = X[params.nbX * params.nbNoeuds_stance + params.nbX*k: params.nbX * params.nbNoeuds_stance + params.nbX*(k + 1)]
    G.append(X[params.nbX * params.nbNoeuds_stance + params.nbX*(k + 1): params.nbX * params.nbNoeuds_stance + params.nbX*(k + 2)] - int_RK4(ffcn_no_contact, params, Xk, Uk, P))
    # G.append(X[params.nbX * params.nbNoeuds_stance + params.nbX*(k + 1): params.nbX * params.nbNoeuds_stance + params.nbX*(k + 2)] - int_RK4_swing(params.T_swing, params.nbNoeuds_swing, params.nkutta,  Xk, Uk))

    # OBJECTIVE FUNCTION
    Jm += fcn_objective_markers(params.wMa, params.wMt, Xk[: params.nbQ], M_real_swing[:, :, k], 'swing')               # tracking marker
    Je += fcn_objective_emg(params.wU, Uk, U_real_swing[:, k])                                                          # tracking emg
    Ja += fcn_objective_activation(params.wL, Uk)                                                                       # min muscular activation

# ----------------------------- Contraintes ----------------------------------------------------------------------------
# égalité
lbg = [0] * params.nbX * params.nbNoeuds
ubg = [0] * params.nbX * params.nbNoeuds

# ----------------------------- Bounds on w ----------------------------------------------------------------------------
# activation - excitation musculaire
min_A = 1e-3                        # 0 exclusif
max_A = 1
lowerbound_u = [min_A] * params.nbMus + [-1000] + [-2000] + [-200]
upperbound_u = [max_A] * params.nbMus + [1000]  + [2000]  + [200]
lbu = (lowerbound_u) * params.nbNoeuds
ubu = (upperbound_u) * params.nbNoeuds

# q et dq
lowerbound_x = [-10, -0.5, -np.pi/4, -np.pi/9, -np.pi/2, -np.pi/3, 0.5, -0.5, -1.7453, -5.2360, -5.2360, -5.2360]
upperbound_x = [10, 1.5, np.pi/4, np.pi/3, 0.0873, np.pi/9, 1.5, 0.5, 1.7453, 5.2360, 5.2360, 5.2360]
lbX   = (lowerbound_x) * (params.nbNoeuds + 1)
ubX   = (upperbound_x) * (params.nbNoeuds + 1)

# parameters
min_pg = 1
min_p  = 0.2
max_p  = 5
max_pg = 1
lbp = [min_pg] + [min_p] * params.nbMus
ubp = [max_pg] + [max_p] * params.nbMus


lbx = vertcat(lbu, lbX, lbp)
ubx = vertcat(ubu, ubX, ubp)

# ----------------------------- Initial guess --------------------------------------------------------------------------
init_A = 0.1

# CONTROL
u0                    = np.zeros((params.nbU, params.nbNoeuds))
u0[: params.nbMus, :] = load_initialguess_muscularExcitation(np.hstack([U_real_stance, U_real_swing]))
# u0[: nbMus, :]   = np.zeros((nbMus, nbNoeuds)) + init_A
u0[params.nbMus + 0, :] = [0] * params.nbNoeuds                                  # pelvis forces
u0[params.nbMus + 1, :] = [-500] * params.nbNoeuds_stance + [0] * params.nbNoeuds_swing
u0[params.nbMus + 2, :] = [0] * params.nbNoeuds

# STATE
q0_stance = load_initialguess_q(params, 'stance')
q0_swing  = load_initialguess_q(params, 'swing')
q0        = np.hstack([q0_stance, q0_swing])
dq0       = np.gradient(q0)
dq0       = dq0[0]
# dq0       = np.zeros((nbQ, (nbNoeuds + 1)))

X0                 = np.zeros((params.nbX, (params.nbNoeuds + 1)))
X0[:params.nbQ, :] = q0
X0[params.nbQ: 2 * params.nbQ, :] = dq0

# PARAMETERS
p0 = [1] + [1] * params.nbMus

w0 = vertcat(vertcat(*u0.T), vertcat(*X0.T), p0)

# ----------------------------- Save txt -------------------------------------------------------------------------------
save_GRF_real(params, GRF_real)
save_Markers_real(params, M_real)
save_EMG_real(params, U_real)
save_params(params)
save_bounds(params, lbx, ubx)
save_initialguess(params, u0, X0, p0)


# ----------------------------- Callback -------------------------------------------------------------------------------
mycallback = AnimateCallback('mycallback', (params.nbU * params.nbNoeuds + params.nbX * (params.nbNoeuds + 1) + params.nP), (params.nbX * params.nbNoeuds), 0)

def print_callback(callback_data):
    print('NEW THREAD print thread')

    name_subject = params.name_subject
    save_dir = params.save_dir
    filename_param = name_subject + '_soldata.txt'
    f = open(save_dir + filename_param, 'a')
    f.write('SOLUTION CALLBACK\n\n')
    np.savetxt(f, w0, delimiter='\n')
    f.close()
    time.sleep(0.001)

    while True:
        if callback_data.update_sol:
            callback_data.update_sol = False
            data = callback_data.sol_data
            f = open(save_dir + filename_param, 'a')
            np.savetxt(f, data, delimiter='\n')
            f.write('\n\n')
            f.close()
        time.sleep(0.001)

# def plot_callback(callback_data):
#     print('NEW THREAD plot thread')
#     # fig, ax = plt.subplots()
#     # ax.plot(np.linspace(0, 10, 10), np.zeros((10)))
#     # ax.set_title('hope it works')
#     # plt.show(block=False)
#     # plt.pause(0.01)
#
#     while True:
#         if callback_data.update_sol:
#             print('NEW DATA plot\n')
#             data = callback_data.sol_data
#             print(str(data[0]) + '\n')
#             # ax.set_ydata(np.random.rand(10))
#             callback_data.update_sol = False
#         # plt.draw()
#         # plt.pause(.001)
#         time.sleep(0.001)
#
# # _thread.start_new_thread(print_callback, (mycallback,))
# plot_thread = threading.Thread(name = 'plot_data', target = plot_callback, args = (mycallback, ))                      # new thread
# plot_thread.start()                                                                                                     # start new thread

print_thread = threading.Thread(name = 'plot_data', target = print_callback, args = (mycallback, ))                      # new thread
print_thread.start()                                                                                                     # start new thread

# ----------------------------- Solver ---------------------------------------------------------------------------------
w = vertcat(U, X, p)
J = Ja + Je + Jm + JR

nlp    = {'x': w, 'f': J, 'g': vertcat(*G)}
opts   = {"ipopt.tol": 1e-2, "ipopt.linear_solver": "ma57", "ipopt.hessian_approximation":"limited-memory", "iteration_callback": mycallback}
solver = nlpsol("solver", "ipopt", nlp, opts)

start_opti = time.time()
res = solver(lbg = lbg,
             ubg = ubg,
             lbx = lbx,
             ubx = ubx,
             x0  = w0)


# RESULTS
stop_opti = time.time() - start_opti
print('Time to solve : ' + str(stop_opti))

# plot_thread.join()

sol_U  = res["x"][:params.nbU * params.nbNoeuds]
sol_X  = res["x"][params.nbU * params.nbNoeuds: -params.nP]
sol_p  = res["x"][-params.nP:]

sol_q  = [sol_X[0::params.nbX], sol_X[1::params.nbX], sol_X[2::params.nbX], sol_X[3::params.nbX], sol_X[4::params.nbX], sol_X[5::params.nbX]]
sol_dq = [sol_X[6::params.nbX], sol_X[7::params.nbX], sol_X[8::params.nbX], sol_X[9::params.nbX], sol_X[10::params.nbX], sol_X[11::params.nbX]]
sol_a  = [sol_U[0::params.nbU], sol_U[1::params.nbU], sol_U[2::params.nbU], sol_U[3::params.nbU], sol_U[4::params.nbU], sol_U[5::params.nbU], sol_U[6::params.nbU],
         sol_U[7::params.nbU], sol_U[8::params.nbU], sol_U[9::params.nbU], sol_U[10::params.nbU], sol_U[11::params.nbU], sol_U[12::params.nbU], sol_U[13::params.nbU],
         sol_U[14::params.nbU], sol_U[15::params.nbU], sol_U[16::params.nbU]]
sol_F  = [sol_U[17::params.nbU], sol_U[18::params.nbU], sol_U[19::params.nbU]]

nbNoeuds_phase = [params.nbNoeuds_stance, params.nbNoeuds_swing]
T_phase        = [params.T_stance, params.T_swing]