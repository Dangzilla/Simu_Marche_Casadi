import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Read_MUSCOD import InitialGuess_MUSCOD
from LoadData import *
from Fcn_Affichage import *
from Marche_Fcn_Integration import *
import biorbd

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
a0  = x0[2 * nbQ:, :]
e0  = u0[:nbMus, :]
F0  = u0[nbMus:, :]


# ----------------------------- Dynamic Results MUSCOD -----------------------------------------------------------------
GRF = np.zeros((3, nbNoeuds))  # ground reaction forces
# SET ISOMETRIC FORCES
n_muscle = 0
for nGrp in range(model_stance.nbMuscleGroups()):
    for nMus in range(model_stance.muscleGroup(nGrp).nbMuscles()):
        fiso = model_stance.muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax()
        model_stance.muscleGroup(nGrp).muscle(nMus).characteristics().setForceIsoMax(p0[n_muscle + 1] * fiso)
        model_swing.muscleGroup(nGrp).muscle(nMus).characteristics().setForceIsoMax(p0[n_muscle + 1] * fiso)

constraint = 0

def fcn_dyn_contact(x, u):
    Q  = x[:nbQ]
    dQ = x[nbQ:]
    states = biorbd.VecBiorbdMuscleStateDynamics(model_stance.nbMuscleTotal())
    n_muscle = 0
    for state in states:
        state.setActivation(u[n_muscle])
        n_muscle += 1

    joint_torque    = model_stance.muscularJointTorque(states, q0[:, k], dq0[:, k]).to_array()
    joint_torque[0] = u[nbMus + 0]  # ajout des forces au pelvis
    joint_torque[1] = u[nbMus + 1]
    joint_torque[2] = u[nbMus + 2]

    ddQ = model_stance.ForwardDynamicsConstraintsDirect(Q, dQ, joint_torque)
    return np.hstack([dQ, ddQ.to_array()])


def fcn_dyn_nocontact(x, u):
    Q  = x[:nbQ]
    dQ = x[nbQ:]
    states = biorbd.VecBiorbdMuscleStateDynamics(model_stance.nbMuscleTotal())
    n_muscle = 0
    for state in states:
        state.setActivation(u[n_muscle])
        n_muscle += 1

    joint_torque = model_swing.muscularJointTorque(states, q0[:, k], dq0[:, k]).to_array()
    joint_torque[0] = u[nbMus + 0]  # ajout des forces au pelvis
    joint_torque[1] = u[nbMus + 1]
    joint_torque[2] = u[nbMus + 2]

    ddQ = model_swing.ForwardDynamics(Q, dQ, joint_torque)
    return np.hstack([dQ, ddQ.to_array()])

def int_RK4(fcn, x, u):
    dn = T_stance / nbNoeuds_stance                 # Time step for shooting point
    dt = dn / 5                                     # Time step for iteration
    xj = x
    for i in range(5):
        k1 = fcn(xj, u)
        x2 = xj + (dt/2)*k1
        k2 = fcn(x2, u)
        x3 = xj + (dt/2)*k2
        k3 = fcn(x3, u)
        x4 = xj + dt*k3
        k4 = fcn(x4, u)

        xj += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return xj

def int_euler(fcn, x, u):
    dn = T / nbNoeuds                                                        # Time step for shooting point
    dt = dn / 20                                                             # Time step for iteration
    xj = x
    for j in range(20):
        dxj = fcn(x, u)
        xj += dt * dxj                                                       # xj = x(t+dt)
    return xj


c = 0
for k in range(nbNoeuds_stance):
    xk  = np.hstack([q0[:, k], dq0[:, k]])
    xk1 = np.hstack([q0[:, k + 1], dq0[:, k + 1]])
    uk  = np.hstack([a0[:, k], F0[:, k]])

    xk1_int_rk4 = int_RK4(fcn_dyn_contact, xk, uk)
    c += xk1 - xk1_int_rk4

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

# # plot integrale
# x_int = np.zeros((nbX, 10))
# x_int[:, 0] = np.hstack([q0[:, 0], dq0[:, 0]])
# T = T/10
# u_int = np.hstack([a0[:, 0], F0[:, 0]])
# for k in range(9):
#      x_int[:, k + 1] = int_RK4(fcn_dyn_contact, x_int[:, k], u_int)


# GROUND REACTION FORCES
diff_F = (GRF_real - GRF) * (GRF_real - GRF)

# JOINT POSITIONS
plt.figure(1)
t_stance = np.linspace(0, T_phase[0], nbNoeuds_phase[0])
t_swing = t_stance[-1] + np.linspace(0, T_phase[1], nbNoeuds_phase[1])
t = np.hstack([t_stance, t_swing])
t = np.hstack([t, t[-1] + (t[-1] - t[-2])])

plt.subplot(231)
plt.title('Pelvis_Trans_X')
plt.plot(t, q0[0, :])
plt.plot([T_stance, T_stance], [min(q0[0, :]), max(q0[0, :])], 'k:')
plt.xlabel('time (s)')
plt.ylabel('position (m)')

plt.subplot(232)
plt.title('Pelvis_Trans_Y')
plt.plot(t, q0[1, :])
plt.plot([T_stance, T_stance], [min(q0[1, :]), max(q0[1, :])], 'k:')
plt.xlabel('time (s)')
plt.ylabel('position (m)')

plt.subplot(233)
plt.title('Pelvis_Rot_Z')
plt.plot(t, q0[2, :]*180/np.pi)
plt.plot([T_stance, T_stance], [min(q0[2, :]*180/np.pi), max(q0[2, :]*180/np.pi)], 'k:')
plt.xlabel('time (s)')
plt.ylabel('angle (deg)')

plt.subplot(234)
plt.title('R_Hip_Rot_Z')
plt.plot(t, q0[3, :]*180/np.pi)
plt.plot([T_stance, T_stance], [min(q0[3, :]*180/np.pi), max(q0[3, :]*180/np.pi)], 'k:')
plt.xlabel('time (s)')
plt.ylabel('angle (deg)')

plt.subplot(235)
plt.title('R_Knee_Rot_Z')
plt.plot(t, q0[4, :]*180/np.pi)
plt.plot([T_stance, T_stance], [min(q0[4, :]*180/np.pi), max(q0[4, :]*180/np.pi)], 'k:')
plt.xlabel('time (s)')
plt.ylabel('angle (deg)')

plt.subplot(236)
plt.title('R_Ankle_Rot_Z')
plt.plot(t, q0[5, :]*180/np.pi)
plt.plot([T_stance, T_stance], [min(q0[5, :]*180/np.pi), max(q0[5, :]*180/np.pi)], 'k:')
plt.xlabel('time (s)')
plt.ylabel('angle (deg)')



# MARKERS -- Heatmap
M_simu = np.zeros((3, nbMarker, nbNoeuds + 1))
for n_st in range(nbNoeuds_phase[0]):
    for nMark in range(nbMarker):
        M_simu[:, nMark, n_st] = model_stance.marker(q0[:, n_st], nMark).to_array()
for n_sw in range(nbNoeuds_phase[1] + 1):
    for nMark in range(nbMarker):
        M_simu[:, nMark, (nbNoeuds_phase[0] + n_sw)] = model_swing.marker(q0[:, (nbNoeuds_phase[0] + n_sw)], nMark).to_array()


Labels_M = ["L_IAS", "L_IPS", "R_IPS", "R_IAS", "R_FTC",
            "R_Thigh_Top", "R_Thigh_Down", "R_Thigh_Front", "R_Thigh_Back", "R_FLE", "R_FME",
            "R_FAX", "R_TTC", "R_Shank_Top", "R_Shank_Down", "R_Shank_Front", "R_Shank_Tibia", "R_FAL", "R_TAM",
            "R_FCC", "R_FM1", "R_FMP1", "R_FM2", "R_FMP2", "R_FM5", "R_FMP5"]
node    = np.linspace(0, nbNoeuds, nbNoeuds, dtype = int)

diff_M = (M_simu - M_real)*(M_simu - M_real)

fig4, ax = plt.subplots()
im       = ax.imshow(diff_M[0, :, :])

# Create labels
ax.set_xticks(np.arange(len(node)))
ax.set_yticks(np.arange(len(Labels_M)))
ax.set_xticklabels(node)
ax.set_yticklabels(Labels_M)
ax.set_title('Markers differences')

# Create grid
ax.set_xticks(np.arange(diff_M[0, :, :].shape[1] + 1) - .5, minor=True)
ax.set_yticks(np.arange(diff_M[0, :, :].shape[0] + 1) - .5, minor=True)
ax.grid(which="minor", color="k", linestyle='-', linewidth=0.2)
ax.tick_params(which="minor", bottom=False, left=False)

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('squared differences', rotation=-90, va="bottom")


# MUSCULAR ACTIVATIONS
plt.figure(2)
Labels = ['GLUT_MAX1', 'GLUT_MAX2', 'GLUT_MAX3', 'GLUT_MED1', 'GLUT_MED2', 'GLUT_MED3', 'R_SEMIMEM', 'R_SEMITEN',
          'R_BI_FEM_LH', 'R_RECTUS_FEM', 'R_VAS_MED', 'R_VAS_INT', 'R_VAS_LAT', 'R_GAS_MED', 'R_GAS_LAT',
          'R_SOLEUS', 'R_TIB_ANT']

nMus_emg = 9
for nMus in range(nbMus):
    plt.subplot(6, 3, nMus + 1)
    plt.title(Labels[nMus])
    plt.plot([T_stance, T_stance], [0, 1], 'k:')

    if nMus == 1 or nMus == 2 or nMus == 3 or nMus == 5 or nMus == 6 or nMus == 11 or nMus == 12:
        plt.plot(t, a0[nMus, :], 'b')
    else:
        plt.plot(t, a0[nMus, :], 'b')
        plt.plot(t, U_real[nMus_emg, :], 'r-')
        nMus_emg -= 1

# EMG -- HeatMap
Labels_emg = ['GLUT_MAX1', 'GLUT_MAX2', 'GLUT_MAX3', 'GLUT_MED1', 'GLUT_MED2', 'GLUT_MED3', 'R_SEMIMEM', 'R_SEMITEN',
              'R_BI_FEM_LH', 'R_RECTUS_FEM', 'R_VAS_MED', 'R_VAS_INT', 'R_VAS_LAT', 'R_GAS_MED', 'R_GAS_LAT',
              'R_SOLEUS', 'R_TIB_ANT']
node       = np.linspace(0, nbNoeuds, nbNoeuds, dtype = int)

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

diff_U = (U_emg - U_real[:, :-1])*(U_emg - U_real[:, :-1])

fig, ax = plt.subplots()
im_emg  = ax.imshow(diff_U[:, :], cmap=plt.get_cmap('YlOrRd'))

# Create labels
ax.set_xticks(np.arange(len(node)))
ax.set_yticks(np.arange(len(Labels_emg)))
ax.set_xticklabels(node)
ax.set_yticklabels(Labels_emg)
ax.set_title('Muscular activations differences')

# Create grid
ax.set_xticks(np.arange(diff_U[:, :].shape[1] + 1) - .5, minor=True)
ax.set_yticks(np.arange(diff_U[:, :].shape[0] + 1) - .5, minor=True)
ax.grid(which="minor", color="k", linestyle='-', linewidth=0.2)
ax.tick_params(which="minor", bottom=False, left=False)

# Create colorbar
cbar = ax.figure.colorbar(im_emg, ax=ax)
cbar.ax.set_ylabel('squared differences ', rotation=-90, va="bottom")

# PELVIS FORCES
plt.figure(3)
plt.subplot(311)
plt.title('Force Pelvis TX')
plt.plot([0, t[-1]], [-1000, -1000], 'k--')  # lower bound
plt.plot([0, t[-1]], [1000, 1000], 'k--')    # upper bound
for n in range(nbNoeuds_phase[0] + nbNoeuds_phase[1] -1):
    plt.plot([t[n], t[n+1], t[n+1]], [F0[0, n], F0[0, n], F0[0, n + 1]], 'b')

plt.subplot(312)
plt.title('Force Pelvis TY')
plt.plot([0, t[-1]], [-2000, -2000], 'k--')  # lower bound
plt.plot([0, t[-1]], [2000, 2000], 'k--')    # upper bound
for n in range(nbNoeuds_phase[0] + nbNoeuds_phase[1] -1):
    plt.plot([t[n], t[n+1], t[n+1]], [F0[1, n], F0[1, n], F0[1, n + 1]], 'b')

plt.subplot(313)
plt.title('Force Pelvis RZ')
plt.plot([0, t[-1]], [-200, -200], 'k--')  # lower bound
plt.plot([0, t[-1]], [200, 200], 'k--')    # upper bound
for n in range(nbNoeuds_phase[0] + nbNoeuds_phase[1] -1):
    plt.plot([t[n], t[n+1], t[n+1]], [F0[2, n], F0[2, n], F0[2, n + 1]], 'b')


# CONVERGENCE
Jm = wMa * sum(diff_M[0, :, :]) + wMa * sum(diff_M[2, :, :])
Je = wU * (sum(diff_U))
Ja = wL * (np.dot(a0[1, :], a0[1, :].T)) + wL * (np.dot(a0[2, :],a0[2, :].T)) + wL * (np.dot(a0[3, :],a0[3, :].T)) + wL * (np.dot(a0[5, :],a0[5, :].T)) + wL * (np.dot(a0[6, :],a0[6, :].T)) + wL * (np.dot(a0[11, :],a0[11, :].T)) + wL * (np.dot(a0[12, :],a0[12, :].T))
JR = wR * (sum(diff_F))

plt.draw()
plt.show()


