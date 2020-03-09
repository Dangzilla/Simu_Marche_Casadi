import biorbd
from ezc3d import c3d
from casadi import *
from pylab import *
import numpy as np
import time
from Marche_Fcn_Integration import *
from Fcn_Objective import *
from LoadData import *
from Fcn_Affichage import *
from Fcn_InitialGuess import *
from Read_MUSCOD import *

# SET MODELS
model_swing  = biorbd.Model('/home/leasanchez/programmation/Marche_Florent/ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod')
model_stance = biorbd.Model('/home/leasanchez/programmation/Marche_Florent/ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod')

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


# ----------------------------- States & controls ----------------------------------------------------------------------
# CONTROL
u  = MX.sym("u", nbU)
e  = u[:nbMus]                                         # muscular excitation
F  = u[nbMus:]                                         # articular torque

# PARAMETERS
p  = MX.sym("p", nP)                                   # maximal isometric force adjustment

# STATE
x  = MX.sym("x", nbX)
q  = x[:nbQ]                                           # generalized coordinates
dq = x[nbQ: 2*nbQ]                                     # velocities


# ----------------------------- Load Data ------------------------------------------------------------------------------
# LOAD MEASUREMENT DATA FROM C3D FILE
file        = '/home/leasanchez/programmation/Simu_Marche_Casadi/DonneesMouvement/equincocont01_out.c3d'
kalman_file = '/home/leasanchez/programmation/Simu_Marche_Casadi/DonneesMouvement/equincocont01_out_MOD5000_leftHanded_GenderF_Florent_.Q2'

# ground reaction forces
[GRF_real, T, T_stance] = load_data_GRF(file, nbNoeuds_stance, nbNoeuds_swing, 'cycle')
T_swing                 = T - T_stance                                                                # gait cycle time
T_phase                 = [T_stance, T_swing]

# marker
M_real_stance = load_data_markers(file, T_stance, nbNoeuds_stance, nbMarker, 'stance')
M_real_swing  = load_data_markers(file, T_swing, nbNoeuds_swing, nbMarker, 'swing')
M_real        = [np.hstack([M_real_stance[0, :, :], M_real_swing[0, :, :]]), np.hstack([M_real_stance[1, :, :], M_real_swing[1, :, :]]), np.hstack([M_real_stance[2, :, :], M_real_swing[2, :, :]])]


# muscular excitation
U_real_swing  = load_data_emg(file, T_swing, nbNoeuds_swing, nbMus, 'swing')
U_real_stance = load_data_emg(file, T_stance, nbNoeuds_stance, nbMus, 'stance')
U_real        = np.hstack([U_real_stance, U_real_swing])

# EXTRACT INITIAL MAXIMAL ISOMETRIC FORCES FROM THE MODEL
FISO0 = np.zeros(nbMus)
n_muscle = 0
for nGrp in range(model_stance.nbMuscleGroups()):
    for nMus in range(model_stance.muscleGroup(nGrp).nbMuscles()):
        FISO0[n_muscle] = model_stance.muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax()
        n_muscle += 1

nkutta = 4                                             # number of iteration for integration

# ----------------------------- Weighting factors ----------------------------------------------------------------------
wL  = 1                                                # activation
wMa = 30                                               # anatomical marker
wMt = 50                                               # technical marker
wU  = 1                                                # excitation
wR  = 0.05                                             # ground reaction




# ----------------------------- Movement -------------------------------------------------------------------------------
U = MX.sym("U", nbU*nbNoeuds)                          # controls
X = MX.sym("X", nbX*(nbNoeuds+1))                      # states
G = []                                                 # equality constraints
Ja = 0                                                 # objective function for muscle activation
Jm = 0                                                 # objective function for markers
Je = 0                                                 # objective function for EMG
JR = 0                                                 # objective function for ground reactions

M_simu = np.zeros((nbMarker, nbNoeuds))
# ------------ PHASE 1 : Stance phase
# FIND THE PARAMETERS P OPTIMISING THE MAXIMUM ISOMETRIC FORCES -- MODEL STANCE
Set_forceISO_max = external('libforce_iso_max_stance', 'libforce_iso_max_stance.so',{'enable_fd':True})
forceISO         = p[0]*p[1:]*FISO0
Set_forceISO_max(forceISO)

for k in range(nbNoeuds_stance):
    # DYNAMIQUE
    Uk = U[nbU*k: nbU*(k + 1)]
    G.append(X[nbX*(k + 1): nbX*(k + 2)] - int_RK4_stance(T_stance, nbNoeuds_stance, nkutta,  X[nbX*k: nbX*(k + 1)], Uk))

    # OBJECTIVE FUNCTION
    # Tracking
    JR += fcn_objective_GRF(wR, X[nbX*k: nbX*(k + 1)], Uk, GRF_real[:, k])                              # Ground Reaction --> stance
    Jm += fcn_objective_markers(wMa, wMt, X[nbX*k: nbX*k + nbQ], M_real_stance[:, :, k], 'stance')      # Marker
    Je += fcn_objective_emg(wU, Uk, U_real_stance[:, k])                                                # EMG

    # Activations
    Ja += fcn_objective_activation(wL, Uk)                                                                 # Muscle activations (no EMG)


# ------------ PHASE 2 : Swing phase
# FIND THE PARAMETERS P OPTIMISING THE MAXIMUM ISOMETRIC FORCES -- MODEL SWING
Set_forceISO_max_swing = external('libforce_iso_max', 'libforce_iso_max.so',{'enable_fd':True})
Set_forceISO_max_swing(forceISO)

for k in range(nbNoeuds_swing):
    # DYNAMIQUE
    Uk = U[nbU*nbNoeuds_stance + nbU*k: nbU * nbNoeuds_stance + nbU*(k + 1)]
    G.append(X[nbX * nbNoeuds_stance + nbX*(k + 1): nbX*nbNoeuds_stance + nbX*(k + 2)] - int_RK4_swing(T_swing, nbNoeuds_swing, nkutta,  X[nbX*nbNoeuds_stance + nbX*k: nbX*nbNoeuds_stance + nbX*(k + 1)], Uk))

    # OBJECTIVE FUNCTION
    # Tracking
    Jm += fcn_objective_markers(wMa, wMt, X[nbX*nbNoeuds_stance + nbX*k: nbX*nbNoeuds_stance + nbX*k + nbQ], M_real_swing[:, :, k], 'swing')   # marker
    Je += fcn_objective_emg(wU, Uk, U_real_swing[:, k])                                                                                        # emg
    # Activations
    Ja += fcn_objective_activation(wL, Uk)

# ----------------------------- Contraintes ----------------------------------------------------------------------------
# égalité
lbg = [0]*nbX*nbNoeuds
ubg = [0]*nbX*nbNoeuds
# contrainte cyclique???

# ----------------------------- Bounds on w ----------------------------------------------------------------------------
# activation - excitation musculaire
min_A = 1e-3                        # 0 exclusif
max_A = 1

lbu = ([min_A]*nbMus + [-1000] + [-2000] + [-200])*nbNoeuds
ubu = ([max_A]*nbMus + [1000]  + [2000]  + [200])*nbNoeuds

# q et dq
min_Q = -50
max_Q = 50
lbX   = ([min_Q]*2*nbQ )*(nbNoeuds + 1)
ubX   = ([max_Q]*2*nbQ)*(nbNoeuds + 1)

# parameters
min_pg = 1
min_p  = 0.2
max_p  = 5
max_pg = 1
lbp = [min_pg] + [min_p]*nbMus
ubp = [max_pg] + [max_p]*nbMus


lbx = vertcat(lbu, lbX, lbp)
ubx = vertcat(ubu, ubX, ubp)

# ----------------------------- Initial guess --------------------------------------------------------------------------
muscod_file  = '/home/leasanchez/programmation/Marche_Florent/ResultatsSimulation/equincocont01_out/RES/ANsWER_gaitCycle_works4.txt'
[u0, x0, p0] = InitialGuess_MUSCOD(muscod_file, nbQ, nbMus, nP, nbNoeuds_phase, T_phase, U_real)
# re interpretation based on state and control changes (ie activation in control instead of state)
q0  = x0[: nbQ, :]
dq0 = x0[nbQ: 2 * nbQ, :]
a0  = x0[2 * nbQ:, :]
e0  = u0[:nbMus, :]
F0  = u0[nbMus:, :]

u0_casadi = np.vstack([a0[:, :-1], F0])
u0_casadi = vertcat(*u0_casadi.T)
x0_casadi = np.vstack([q0, dq0])
x0_casadi = vertcat(*x0_casadi.T)


# SAVE MUSCOD RESULTS
txt_file = '/home/leasanchez/programmation/Simu_Marche_Casadi/InitialGuess/Results_MUSCOD.txt'
f        = open(txt_file, 'a')
f.write("Results from MUSCOD \n\n")
f.write("from " + muscod_file + "\n\n\n")
f.write("CONTROLS \n\n")
for n in range(nbNoeuds):
    f.write("u(:, " + str(n) + ") \n")
    np.savetxt(f, u0[:, n], delimiter='\n')
    f.write("\n")
f.write("\n\n\n STATES \n\n")
for n in range(nbNoeuds + 1):
    f.write("x(:, " + str(n) + ") \n")
    np.savetxt(f, x0[:, n], delimiter='\n')
    f.write("\n")
f.write("\n\n\n PARAMETERS \n\n")
np.savetxt(f, p0, delimiter='\n')
f.close()


# SAVE INITIAL GUESS
txt_file2 = '/home/leasanchez/programmation/Simu_Marche_Casadi/InitialGuess/InitialGuess_from_MUSCOD.txt'
f2        = open(txt_file2, 'a')
f2.write("Initial guess from MUSCOD \n\n")
f2.write("from " + muscod_file + "\n\n\n")
f2.write("CONTROLS \n\n")
for n in range(nbNoeuds):
    f2.write("u(:, " + str(n) + ") \n")
    np.savetxt(f2, np.hstack([x0[2*nbQ:, n], u0[nbMus:, n]]), delimiter='\n')
    f2.write("\n")
f2.write("\n\n\n STATES \n\n")
for n in range(nbNoeuds + 1):
    f2.write("x(:, " + str(n) + ") \n")
    np.savetxt(f2, x0[:2*nbQ, n], delimiter='\n')
    f2.write("\n")
f2.write("\n\n\n PARAMETERS \n\n")
np.savetxt(f2, p0, delimiter='\n')
f2.close()

# ----------------------------- Solver ---------------------------------------------------------------------------------
w = vertcat(U, X, p)
J = Ja + Je + Jm + JR

nlp = {'x': w, 'f': J, 'g': vertcat(*G)}
opts = {"ipopt.tol": 1e-1, "ipopt.linear_solver": "ma57", "ipopt.hessian_approximation":"limited-memory"}
solver = nlpsol("solver", "ipopt", nlp, opts)

start_opti = time.time()
print('Start optimisation : ' + str(start_opti))
res = solver(lbg = lbg,
             ubg = ubg,
             lbx = lbx,
             ubx = ubx,
             x0  = vertcat(u0_casadi, x0_casadi, p0))


# RESULTS
stop_opti = time.time() - start_opti
print('Time to solve : ' + str(stop_opti))
save()

sol_U  = res["x"][:nbU * nbNoeuds]
sol_X  = res["x"][nbU * nbNoeuds: -nP]
sol_p  = res["x"][-nP:]

sol_q = [np.array(sol_X[0::nbX]).squeeze(), np.array(sol_X[1::nbX]).squeeze(), np.array(sol_X[2::nbX]).squeeze(),
         np.array(sol_X[3::nbX]).squeeze(), np.array(sol_X[4::nbX]).squeeze(), np.array(sol_X[5::nbX]).squeeze()]

sol_dq = [np.array(sol_X[6::nbX]).squeeze(), np.array(sol_X[7::nbX]).squeeze(), np.array(sol_X[8::nbX]).squeeze(),
          np.array(sol_X[9::nbX]).squeeze(), np.array(sol_X[10::nbX]).squeeze(), np.array(sol_X[11::nbX]).squeeze()]

sol_a = [np.array(sol_U[0::nbU]).squeeze(), np.array(sol_U[1::nbU]).squeeze(), np.array(sol_U[2::nbU]).squeeze(),
         np.array(sol_U[3::nbU]).squeeze(), np.array(sol_U[4::nbU]).squeeze(), np.array(sol_U[5::nbU]).squeeze(),
         np.array(sol_U[6::nbU]).squeeze(), np.array(sol_U[7::nbU]).squeeze(), np.array(sol_U[8::nbU]).squeeze(),
         np.array(sol_U[9::nbU]).squeeze(), np.array(sol_U[10::nbU]).squeeze(), np.array(sol_U[11::nbU]).squeeze(),
         np.array(sol_U[12::nbU]).squeeze(), np.array(sol_U[13::nbU]).squeeze(), np.array(sol_U[14::nbU]).squeeze(),
         np.array(sol_U[15::nbU]).squeeze(), np.array(sol_U[16::nbU]).squeeze()]

sol_F = [np.array(sol_U[17::nbU]).squeeze(), np.array(sol_U[18::nbU]).squeeze(), np.array(sol_U[19::nbU]).squeeze()]


# SAVE RESULTS
txt_file_res = '/home/leasanchez/programmation/Simu_Marche_Casadi/InitialGuess/ResultsCasadi_from_MUSCOD.txt'
f_res        = open(txt_file_res, 'a')
f_res.write("Results from MUSCOD initial guess \n\n")
f_res.write("from " + muscod_file + "\n\n\n")
f_res.write("CONTROLS \n\n")
for n in range(nbNoeuds):
    f_res.write("u(:, " + str(n) + ") \n")
    np.savetxt(f_res, np.hstack([sol_a[:, n], sol_F[:, n]]), delimiter='\n')
    f_res.write("\n")
f_res.write("\n\n\n STATES \n\n")
for n in range(nbNoeuds + 1):
    f_res.write("x(:, " + str(n) + ") \n")
    np.savetxt(f_res, np.hstack([sol_q[:, n], sol_dq[:, n]]), delimiter='\n')
    f_res.write("\n")
f_res.write("\n\n\n PARAMETERS \n\n")
np.savetxt(f_res, p, delimiter='\n')
f_res.close()

nbNoeuds_phase = [nbNoeuds_stance, nbNoeuds_swing]
T_phase        = [T_stance, T_swing]