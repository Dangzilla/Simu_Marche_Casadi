from casadi import *
from matplotlib import pyplot as plt
import biorbd
import numpy as np

# add classes
from Define_parameters import Parameters

# add fcn
from LoadData import load_data_markers, load_data_emg, load_data_GRF
from Fcn_InitialGuess import load_initialguess_muscularExcitation, load_initialguess_q
from Marche_Fcn_Integration import int_RK4_swing, int_RK4_stance, int_RK4
from Fcn_forward_dynamic import ffcn_contact, ffcn_no_contact
from Fcn_Objective import fcn_objective_activation, fcn_objective_emg, fcn_objective_markers, fcn_objective_GRF

# SET PARAMETERS
params = Parameters()

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
Ja = 0                                                 # objective function for muscle activation
Jm = 0                                                 # objective function for markers
Je = 0                                                 # objective function for EMG
JR = 0                                                 # objective function for ground reactions
constraint = 0                                         # constraints
q_int = np.zeros((params.nbQ, params.nbNoeuds))
dq_int = np.zeros((params.nbQ, params.nbNoeuds))
GRF = np.zeros((3, params.nbNoeuds))                   # ground reaction forces


# ----------------------------- Initial guess --------------------------------------------------------------------------
init_A = 0.1

# CONTROL
u0                    = np.zeros((params.nbU, params.nbNoeuds))
u0[: params.nbMus, :] = load_initialguess_muscularExcitation(np.hstack([U_real_stance, U_real_swing]))
# u0[: nbMus, :]   = np.zeros((nbMus, nbNoeuds)) + init_A
u0[params.nbMus + 0, :] = [0] * params.nbNoeuds                                  # pelvis forces
u0[params.nbMus + 1, :] = [-500] * params.nbNoeuds_stance + [0] * params.nbNoeuds_swing
u0[params.nbMus + 2, :] = [0] * params.nbNoeuds
u0 = vertcat(*u0.T)

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
X0 = vertcat(*X0.T)

# PARAMETERS
p0 = [1] + [1] * params.nbMus


# ------------ PHASE 1 : Stance phase
for k in range(params.nbNoeuds_stance):
    Uk = u0[params.nbU*k : params.nbU*(k + 1)]
    Xk = X0[params.nbX*k: params.nbX*(k + 1)]

    Q           = np.array(Xk[:params.nbQ]).squeeze()  # states
    dQ          = np.array(Xk[params.nbQ:2 * params.nbQ]).squeeze()
    activations = np.array(Uk[: params.nbMus]).squeeze()  # controls
    F           = np.array(Uk[params.nbMus :])

    X_int = int_RK4(ffcn_contact, params, Xk, Uk, p0)
    q_int[:, k] = np.array(X_int[:params.nbQ]).squeeze()
    dq_int[:, k] = np.array(X_int[params.nbQ:2 * params.nbQ]).squeeze()
    constraint += X0[params.nbX*(k + 1): params.nbX*(k + 2)] - X_int

    # DYNAMIQUE
    m = params.model_stance
    states = biorbd.VecBiorbdMuscleStateDynamics(params.nbMus)
    n_muscle = 0
    for state in states:
        state.setActivation(activations[n_muscle])  # Set muscles activations
        n_muscle += 1

    torque    = m.muscularJointTorque(states, Q, dQ).to_array()
    torque[0] = F[0]
    torque[1] = F[1]
    torque[2] = F[2]

    C    = m.getConstraints()
    ddQ  = m.ForwardDynamicsConstraintsDirect(Q, dQ, torque, C)
    GRF[:, k] = C.getForce().to_array()

    # OBJECTIVE FUNCTION
    JR += params.wR * ((GRF[0, k] - GRF_real[1, k]) * (GRF[0, k] - GRF_real[1, k])) + params.wR * ((GRF[2, k] - GRF_real[2, k]) * (GRF[2, k] - GRF_real[2, k]))
    Jm += fcn_objective_markers(params.wMa, params.wMt, Q, M_real_stance[:, :, k], 'stance')
    Je += fcn_objective_emg(params.wU, Uk, U_real_stance[:, k])
    Ja += fcn_objective_activation(params.wL, Uk)

GRF = np.vstack([GRF])
# ------------ PHASE 2 : Swing phase
for k in range(params.nbNoeuds_swing):
    # CONTROL AND STATES
    Uk = u0[params.nbU * params.nbNoeuds_stance + params.nbU*k: params.nbU * params.nbNoeuds_stance + params.nbU*(k + 1)]
    Xk = X0[params.nbX * params.nbNoeuds_stance + params.nbX * k: params.nbX * params.nbNoeuds_stance + params.nbX * (k + 1)]

    Q           = Xk[:params.nbQ]  # states
    dQ          = Xk[params.nbQ:2 * params.nbQ]
    activations = Uk[: params.nbMus]  # controls
    F           = Uk[params.nbMus :]

    X_int = int_RK4(ffcn_no_contact, params, Xk, Uk, p0)
    q_int[:, k] = X_int[:params.nbQ]
    dq_int[:, k] = X_int[params.nbQ:2 * params.nbQ]

    # OBJECTIVE FUNCTION
    Jm += fcn_objective_markers(params.wMa, params.wMt, Q, M_real_swing[:, :, k], 'swing')
    Je += fcn_objective_emg(params.wU, Uk, U_real_swing[:, k])
    Ja += fcn_objective_activation(params.wL, Uk)

# # ----------------------------- Visualization --------------------------------------------------------------------------
# # GROUND REACTION FORCES
# GRF = np.vstack([GRF])
#
# plt.figure(1)
# plt.subplot(211)
# plt.title('Ground reactions forces A/P')
# t_stance = np.linspace(0, T_stance, nbNoeuds_stance)
# t_swing = t_stance[-1] + np.linspace(0, T_swing, nbNoeuds_swing)
# t = np.hstack([t_stance, t_swing])
#
# plt.plot(t, GRF_real[1, :-1], 'b-', alpha=0.5, label = 'real')
# plt.plot(t_stance, GRF[:, 0], 'r+-', label = 'simu')
# plt.legend()
#
# plt.subplot(212)
# plt.title('Ground reactions forces vertical')
# plt.plot(t, GRF_real[2, :-1], 'b-', alpha=0.5, label = 'real')
# plt.plot(t_stance, GRF[:, 2], 'r+-', label = 'simu')
# plt.legend()
#
#
# # JOINT POSITIONS AND VELOCITIES
# plt.figure(3)
# Labels_X = ['Pelvis_X', 'Pelvis_Y', 'Pelvis_Z', 'Hip', 'Knee', 'Ankle']
# tq = np.hstack([t, t[-1] + (t[-1]-t[-2])])
#
# for q in range(nbQ):
#     plt.subplot(2, 6, q + 1)
#     plt.title('Q ' + Labels_X[q])
#     plt.plot(tq, q0[q, :]*180/np.pi)
#     plt.xlabel('time [s]')
#     if q == 0:
#         plt.ylabel('q [°]')
#
#     plt.subplot(2, 6, q + 1 + nbQ)
#     plt.title('dQ ' + Labels_X[q])
#     plt.plot(tq, dq0[q, :]*180/np.pi)
#     plt.xlabel('time [s]')
#     if q == 0:
#         plt.ylabel('dq [°/s]')
#
# # MUSCULAR ACTIVATIONS
# plt.figure(2)
# U_real = np.hstack([U_real_stance, U_real_swing])
# Labels = ['GLUT_MAX1', 'GLUT_MAX2', 'GLUT_MAX3', 'GLUT_MED1', 'GLUT_MED2', 'GLUT_MED3', 'R_SEMIMEM', 'R_SEMITEN', 'R_BI_FEM_LH', 'R_RECTUS_FEM', 'R_VAS_MED', 'R_VAS_INT', 'R_VAS_LAT', 'R_GAS_MED', 'R_GAS_LAT', 'R_SOLEUS', 'R_TIB_ANT']
# nMus_emg = 9
#
# for nMus in range(nbMus):
#     plt.subplot(5, 4, nMus + 1)
#     plt.title(Labels[nMus])
#     plt.plot([0, t[-1]], [0, 0], 'k--')  # lower bound
#     plt.plot([0, t[-1]], [1, 1], 'k--')  # upper bound
#
#     if nMus == 1 or nMus == 2 or nMus == 3 or nMus == 5 or nMus == 6 or nMus == 11 or nMus == 12:
#         plt.plot(t, a0[nMus, :], 'r+')
#     else:
#         plt.plot(t, a0[nMus, :], 'r+')
#         plt.plot(t, U_real[nMus_emg, :], 'b-', alpha=0.5)
#         nMus_emg -= 1
#
#
# plt.subplot(5, 4, nbMus + 1)
# plt.title('Pelvis Tx')
# plt.plot(t, f0, 'b+')
#
# plt.subplot(5, 4, nbMus + 2)
# plt.title('Pelvis Ty')
# plt.plot(t, f1, 'b+')
#
#
# plt.subplot(5, 4, nbMus + 3)
# plt.title('Pelvis Rz')
# plt.plot(t, f2, 'b+')


# OBJECTIVE FUNCTION VALUES
J = Ja + Je + Jm + JR
print('Global                 : ' + str(J))
print('activation             : ' + str(Ja))
print('emg                    : ' + str(Je))
print('marker                 : ' + str(Jm))
print('ground reaction forces : ' + str(JR))

# plt.show()