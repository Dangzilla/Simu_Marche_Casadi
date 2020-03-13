import numpy as np
import biorbd
import matplotlib.pyplot as plt
from Read_MUSCOD import InitialGuess_MUSCOD
from LoadData import load_data_GRF, load_data_markers, load_data_emg
from Fcn_plot_results import plot_q_MUSCOD, plot_dq_MUSCOD, plot_markers_heatmap, plot_emg_heatmap, plot_control_MUSCOD, plot_GRF_MUSCOD, plot_markers_result


# SET MODELS
model_swing  = biorbd.Model('/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod')
model_stance = biorbd.Model('/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod')

# ----------------------------- Probleme -------------------------------------------------------------------------------
nbNoeuds_stance = 25                                   # shooting points for stance phase
nbNoeuds_swing  = 25                                   # shooting points for swing phase
nbNoeuds        = nbNoeuds_stance + nbNoeuds_swing     # total shooting points
nbNoeuds_phase  = [nbNoeuds_stance, nbNoeuds_swing]

nbMus     = model_stance.nbMuscleTotal()               # number of muscles
nbQ       = model_stance.nbDof()                       # number of DoFs
nbMarker  = model_stance.nbMarkers()                   # number of markers
nbBody    = model_stance.nbSegment()                   # number of body segments
nbContact = model_stance.nbContacts()                  # number of contact (2 forces --> plan)

nbU      = nbMus + 3                                   # number of controls : muscle activation + articular torque
nbX      = 2*nbQ                                       # number of states : generalized positions + velocities
nP       = nbMus + 1                                   # number of parameters : 1 global + muscles


# ----------------------------- Load Data ------------------------------------------------------------------------------
# LOAD MEASUREMENT DATA FROM C3D FILE
file        = '/home/leasanchez/programmation/Simu_Marche_Casadi/DonneesMouvement/equincocont01_out.c3d'
kalman_file = '/home/leasanchez/programmation/Simu_Marche_Casadi/DonneesMouvement/equincocont01_out_MOD5000_leftHanded_GenderF_Florent_.Q2'

# ground reaction forces
[GRF_real, T, T_stance] = load_data_GRF(file, nbNoeuds_stance, nbNoeuds_swing, 'cycle')
T_swing                 = T - T_stance                                                                # gait cycle time
T_phase                 = [T_stance, T_swing]

# marker
M_real_stance   = load_data_markers(file, T_stance, nbNoeuds_stance, nbMarker, 'stance')
M_real_swing    = load_data_markers(file, T_swing, nbNoeuds_swing, nbMarker, 'swing')
M_real          = np.zeros((3, nbMarker, (nbNoeuds + 1)))
M_real[0, :, :] = np.hstack([M_real_stance[0, :, :], M_real_swing[0, :, :]])
M_real[1, :, :] = np.hstack([M_real_stance[1, :, :], M_real_swing[1, :, :]])
M_real[2, :, :] = np.hstack([M_real_stance[2, :, :], M_real_swing[2, :, :]])


# muscular excitation
U_real_swing  = load_data_emg(file, T_swing, nbNoeuds_swing, nbMus, 'swing')
U_real_stance = load_data_emg(file, T_stance, nbNoeuds_stance, nbMus, 'stance')
U_real        = np.hstack([U_real_stance, U_real_swing])

# force isomax
F_iso0 = np.zeros(nbMus)
n_muscle = 0
for nGrp in range(model_stance.nbMuscleGroups()):
    for nMus in range(model_stance.muscleGroup(nGrp).nbMuscles()):
        F_iso0[n_muscle] = model_stance.muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax()
        n_muscle += 1

# ----------------------------- Weighting factors ----------------------------------------------------------------------
wL  = 1                                                # activation
wMa = 30                                               # anatomical marker
wMt = 50                                               # technical marker
wU  = 1                                                # excitation
wR  = 0.05                                             # ground reaction


# ----------------------------- Load Results MUSCOD --------------------------------------------------------------------
muscod_file  = '/home/leasanchez/programmation/Marche_Florent/ResultatsSimulation/equincocont01_out/RES/ANsWER_gaitCycle_works4.txt'
[u0, x0, p0] = InitialGuess_MUSCOD(muscod_file, nbQ, nbMus, nP, nbNoeuds_phase)

# re interpretation based on state and control changes (ie activation in control instead of state)
q0  = x0[: nbQ, :]
dq0 = x0[nbQ: 2 * nbQ, :]
a0  = x0[2 * nbQ:, :-1]
e0  = u0[:nbMus, :]
F0  = u0[nbMus:, :]


# ----------------------------- Dynamic Results MUSCOD -----------------------------------------------------------------
def int_RK4_nocontact(x, u):
    # fcn ode
    def fcn_dyn_nocontact(x, u):
        m  = biorbd.Model('/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod')
        Q  = x[:nbQ]
        dQ = x[nbQ:]

        # SET ISOMETRIC FORCES
        n_muscle = 0
        for nGrp in range(m.nbMuscleGroups()):
            for nMus in range(m.muscleGroup(nGrp).nbMuscles()):
                m.muscleGroup(nGrp).muscle(nMus).characteristics().setForceIsoMax(p0[n_muscle + 1] * F_iso0[n_muscle])
                n_muscle += 1

        # SET ACTIVATION
        states = biorbd.VecBiorbdMuscleStateDynamics(m.nbMuscleTotal())
        n_muscle = 0
        for state in states:
            state.setActivation(u[n_muscle])
            n_muscle += 1

        # CALCULATE JOINT TORQUE
        joint_torque    = m.muscularJointTorque(states, Q, dQ).to_array()
        joint_torque[0] = u[nbMus + 0]  # ajout des forces au pelvis
        joint_torque[1] = u[nbMus + 1]
        joint_torque[2] = u[nbMus + 2]
        # joint_torque = np.zeros(nbQ)
        # FORWARD DYNAMIQUE
        ddQ = m.ForwardDynamics(Q, dQ, joint_torque)

        return np.hstack([dQ, ddQ.to_array()])

    dn = T_swing / nbNoeuds_swing                   # Time step for shooting point
    dt = dn / 4                                     # Time step for iteration
    xj = x
    for i in range(4):
        k1 = fcn_dyn_nocontact(xj, u)
        x2 = xj + (dt/2)*k1
        k2 = fcn_dyn_nocontact(x2, u)
        x3 = xj + (dt/2)*k2
        k3 = fcn_dyn_nocontact(x3, u)
        x4 = xj + dt*k3
        k4 = fcn_dyn_nocontact(x4, u)

        xj += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return xj

def int_RK4_contact(x, u):
    def fcn_dyn_contact(x, u):
        m  = biorbd.Model('/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod')
        Q  = x[:nbQ]
        dQ = x[nbQ:]

        # SET ISOMETRIC FORCES
        n_muscle = 0
        for nGrp in range(m.nbMuscleGroups()):
            for nMus in range(m.muscleGroup(nGrp).nbMuscles()):
                m.muscleGroup(nGrp).muscle(nMus).characteristics().setForceIsoMax(p0[n_muscle + 1] * F_iso0[n_muscle])
                n_muscle += 1

        # SET ACTIVATION
        states = biorbd.VecBiorbdMuscleStateDynamics(m.nbMuscleTotal())
        n_muscle = 0
        for state in states:
            state.setActivation(u[n_muscle])
            n_muscle += 1

        # CALCULATE JOINT TORQUE
        joint_torque    = m.muscularJointTorque(states, Q, dQ).to_array()
        joint_torque[0] = u[nbMus + 0]  # ajout des forces au pelvis
        joint_torque[1] = u[nbMus + 1]
        joint_torque[2] = u[nbMus + 2]

        # joint_torque = np.zeros(nbQ)
        # FORWARD DYNAMIQUE
        ddQ = m.ForwardDynamicsConstraintsDirect(Q, dQ, joint_torque)
        return np.hstack([dQ, ddQ.to_array()])

    dn = T_stance / nbNoeuds_stance                  # Time step for shooting point
    dt = dn / 10                                     # Time step for iteration
    xj = x
    for i in range(10):
        k1 = fcn_dyn_contact(xj, u)
        x2 = xj + (dt/2)*k1
        k2 = fcn_dyn_contact(x2, u)
        x3 = xj + (dt/2)*k2
        k3 = fcn_dyn_contact(x3, u)
        x4 = xj + dt*k3
        k4 = fcn_dyn_contact(x4, u)

        xj += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return xj


# CONTRAINTES
constraint = np.zeros((2*nbQ, nbNoeuds))

q2 = np.zeros((nbQ, nbNoeuds + 1))
q2[:, 0] = q0[:, 0]

dq2 = np.zeros((nbQ, nbNoeuds + 1))
dq2[:, 0] = dq0[:, 0]

# for k in range(nbNoeuds_stance):
#     xk  = np.hstack([q0[:, k], dq0[:, k]])
#     xk1 = np.hstack([q0[:, k + 1], dq0[:, k + 1]])
#     uk  = np.hstack([a0[:, k], F0[:, k]])
#
#     xk1_int_rk4 = int_RK4_contact(xk, uk)
#     constraint[:, k] = xk1 - xk1_int_rk4
#     q2[:, k + 1] = xk1_int_rk4[:nbQ]
#     dq2[:, k + 1] = xk1_int_rk4[nbQ:]
#
# for k in range(nbNoeuds_swing):
#     xk  = np.hstack([q0[:, nbNoeuds_stance + k], dq0[:, nbNoeuds_stance + k]])
#     xk1 = np.hstack([q0[:, nbNoeuds_stance + k + 1], dq0[:, nbNoeuds_stance + k + 1]])
#     uk  = np.hstack([a0[:, nbNoeuds_stance + k], F0[:, nbNoeuds_stance + k]])
#
#     xk1_int_rk4 = int_RK4_nocontact(xk, uk)
#     constraint[:, nbNoeuds_stance + k] = xk1 - xk1_int_rk4
#     q2[:, nbNoeuds_stance + k + 1] = xk1_int_rk4[:nbQ]
#     dq2[:, nbNoeuds_stance + k + 1] = xk1_int_rk4[nbQ:]

for k in range(nbNoeuds):
    xk  = np.hstack([q0[:, k], dq0[:, k]])
    xk1 = np.hstack([q0[:, k + 1], dq0[:, k + 1]])
    uk  = np.hstack([a0[:, k], F0[:, k]])
    # uk = np.zeros(nbMus + 3)

    if k < nbNoeuds_stance + 1:
        xk1_int_rk4 = int_RK4_contact(xk, uk)
    else:
        xk1_int_rk4 = int_RK4_nocontact(xk, uk)

    constraint[:, k] = xk1 - xk1_int_rk4
    q2[:, k + 1] = xk1_int_rk4[:nbQ]
    dq2[:, k + 1] = xk1_int_rk4[nbQ:]

plot_markers_result(q2, T_phase, nbNoeuds_phase, nbMarker, M_real)

# # JOINT TORQUE
# joint_torque = np.zeros((nbQ, nbNoeuds))
#
# for k in range(nbNoeuds_stance):
#     xk  = np.hstack([q0[:, k], dq0[:, k]])
#     uk  = np.hstack([a0[:, k], F0[:, k]])
#
#     # SET ISOMETRIC FORCES
#     n_muscle = 0
#     for nGrp in range(model_stance.nbMuscleGroups()):
#         for nMus in range(model_stance.muscleGroup(nGrp).nbMuscles()):
#             model_stance.muscleGroup(nGrp).muscle(nMus).characteristics().setForceIsoMax(p0[n_muscle + 1] * F_iso0[n_muscle])
#             n_muscle += 1
#
#     # SET ACTIVATION
#     states = biorbd.VecBiorbdMuscleStateDynamics(model_stance.nbMuscleTotal())
#     n_muscle = 0
#     for state in states:
#         state.setActivation(uk[n_muscle])
#         n_muscle += 1
#
#     # CALCULATE JOINT TORQUE
#     joint_torque[:, k] = model_stance.muscularJointTorque(states, q0[:, k], dq0[:, k]).to_array()
#     joint_torque[0, k] = uk[nbMus + 0]  # ajout des forces au pelvis
#     joint_torque[1, k] = uk[nbMus + 1]
#     joint_torque[2, k] = uk[nbMus + 2]
#
#
# for k in range(nbNoeuds_swing):
#     xk  = np.hstack([q0[:, nbNoeuds_stance + k], dq0[:, nbNoeuds_stance + k]])
#     uk  = np.hstack([a0[:, nbNoeuds_stance + k], F0[:, nbNoeuds_stance + k]])
#
#     # SET ISOMETRIC FORCES
#     n_muscle = 0
#     for nGrp in range(model_swing.nbMuscleGroups()):
#         for nMus in range(model_swing.muscleGroup(nGrp).nbMuscles()):
#             model_swing.muscleGroup(nGrp).muscle(nMus).characteristics().setForceIsoMax(p0[n_muscle + 1] * F_iso0[n_muscle])
#             n_muscle += 1
#
#     # SET ACTIVATION
#     states = biorbd.VecBiorbdMuscleStateDynamics(model_swing.nbMuscleTotal())
#     n_muscle = 0
#     for state in states:
#         state.setActivation(uk[n_muscle])
#         n_muscle += 1
#
#     # CALCULATE JOINT TORQUE
#     joint_torque[:, nbNoeuds_stance + k] = model_swing.muscularJointTorque(states, q0[:, nbNoeuds_stance + k], dq0[:, nbNoeuds_stance + k]).to_array()
#     joint_torque[0, nbNoeuds_stance + k] = uk[nbMus + 0]  # ajout des forces au pelvis
#     joint_torque[1, nbNoeuds_stance + k] = uk[nbMus + 1]
#     joint_torque[2, nbNoeuds_stance + k] = uk[nbMus + 2]
#

# GROUND REACTION FORCES
GRF = np.zeros((3, nbNoeuds))  # ground reaction forces
for k in range(nbNoeuds_stance):
    xk  = np.hstack([q0[:, k], dq0[:, k]])
    xk1 = np.hstack([q0[:, k + 1], dq0[:, k + 1]])
    uk  = np.hstack([a0[:, k], F0[:, k]])

    states   = biorbd.VecBiorbdMuscleStateDynamics(model_stance.nbMuscleTotal())
    n_muscle = 0
    for state in states:
        state.setActivation(a0[n_muscle, k])
        n_muscle += 1

    joint_torque    = model_stance.muscularJointTorque(states, q0[:, k], dq0[:, k]).to_array()
    joint_torque[0] = F0[0, k]                              # ajout des forces au pelvis
    joint_torque[1] = F0[1, k]
    joint_torque[2] = F0[2, k]

    C   = model_stance.getConstraints()
    ddq = model_stance.ForwardDynamicsConstraintsDirect(q0[:, k], dq0[:, k], joint_torque, C)
    GRF[:, k] = C.getForce().to_array()

JR = wR * np.dot((GRF_real[1, :] - GRF[0, :]), (GRF_real[1, :] - GRF[0, :])) + wR * np.dot((GRF_real[2, :] - GRF[2, :]), (GRF_real[2, :] - GRF[2, :]))

# TRACKING MARKERS
M_simu = np.zeros((3, nbMarker, nbNoeuds + 1))
for n_st in range(nbNoeuds_phase[0]):
    for nMark in range(nbMarker):
        M_simu[:, nMark, n_st] = model_stance.marker(q0[:, n_st], nMark).to_array()
for n_sw in range(nbNoeuds_phase[1] + 1):
    for nMark in range(nbMarker):
        M_simu[:, nMark, (nbNoeuds_phase[0] + n_sw)] = model_swing.marker(q0[:, (nbNoeuds_phase[0] + n_sw)],nMark).to_array()

diff_M = (M_simu - M_real) * (M_simu - M_real)
Jm = wMa * np.sum(diff_M[0, :, :]) + wMa * np.sum(diff_M[2, :, :])

# TRACKING EMG
# get only muscles with emg
U_emg = np.zeros(((nbMus - 7), nbNoeuds))
U_emg[0, :] = e0[0, :]
U_emg[1, :] = e0[4, :]
U_emg[2, :] = e0[7, :]
U_emg[3, :] = e0[8, :]
U_emg[4, :] = e0[9, :]
U_emg[5, :] = e0[10, :]
U_emg[6, :] = e0[13, :]
U_emg[7, :] = e0[14, :]
U_emg[8, :] = e0[15, :]
U_emg[9, :] = e0[16, :]

diff_U = (U_emg - U_real) * (U_emg - U_real)
Je = wU * (np.sum(diff_U))

# ACTIVATIONS
Ja = wL * (np.dot(a0[1, :], a0[1, :].T)) + wL * (np.dot(a0[2, :],a0[2, :].T)) + wL * (np.dot(a0[3, :],a0[3, :].T)) + wL * (np.dot(a0[5, :],a0[5, :].T)) + wL * (np.dot(a0[6, :],a0[6, :].T)) + wL * (np.dot(a0[11, :],a0[11, :].T)) + wL * (np.dot(a0[12, :],a0[12, :].T))


# VISUALISATION
# ------ States ----------------------
plot_q_MUSCOD(q0, T_phase, nbNoeuds_phase)
plot_dq_MUSCOD(dq0, T_phase, nbNoeuds_phase)
plot_markers_result(q0, T_phase, nbNoeuds_phase, nbMarker, M_real)

# ------ Control ---------------------
plot_control_MUSCOD(np.vstack([a0, F0]), U_real, nbNoeuds_phase, T_phase)

# ------ Objective function ----------
plot_markers_heatmap(diff_M)
plot_emg_heatmap(diff_U)
plot_GRF_MUSCOD(GRF, GRF_real, nbNoeuds_phase, T_phase)

# PRINT VALUES
print('\n \nGlobal                 : ' + str(Ja + Je + Jm + JR))
print('activation             : ' + str(Ja))
print('emg                    : ' + str(Je))
print('marker                 : ' + str(Jm))
print('ground reaction forces : ' + str(JR))
# print('constraints            : ' + str(sum(constraint)) + '\n')