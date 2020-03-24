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
from Fcn_Affichage import affichage_markers_result

def get_last_n_lines(file_name, N):
    # Create an empty list to keep the track of last N lines
    list_of_lines = []
    # Open file for reading in binary mode
    with open(file_name, 'rb') as read_obj:
        # Move the cursor to the end of the file
        read_obj.seek(0, os.SEEK_END)
        # Create a buffer to keep the last read line
        buffer = bytearray()
        # Get the current position of pointer i.e eof
        pointer_location = read_obj.tell()
        # Loop till pointer reaches the top of the file
        while pointer_location >= 0:
            # Move the file pointer to the location pointed by pointer_location
            read_obj.seek(pointer_location)
            # Shift pointer location by -1
            pointer_location = pointer_location - 1
            # read that byte / character
            new_byte = read_obj.read(1)
            # If the read byte is new line character then it means one line is read
            if new_byte == b'\n':
                # Save the line in list of lines
                list_of_lines.append(buffer.decode()[::-1])
                # If the size of list reaches N, then return the reversed list
                if len(list_of_lines) == N:
                    return list(reversed(list_of_lines))
                # Reinitialize the byte array to save next line
                buffer = bytearray()
            else:
                # If last read character is not eol then add it in buffer
                buffer.extend(new_byte)

        # As file is read completely, if there is still data in buffer, then its first line.
        if len(buffer) > 0:
            list_of_lines.append(buffer.decode()[::-1])

    # return the reversed list
    return list(reversed(list_of_lines))

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


# ----------------------------- Solution from txt file -----------------------------------------------------------------
sol = get_last_n_lines('/home/leasanchez/programmation/Simu_Marche_Casadi/Resultats/equincocont01/equincocont01_soldata.txt', 1633)
sol = np.array(sol[:-3],dtype=float)

U = sol[:(params.nbU * params.nbNoeuds)]
X = sol[(params.nbU * params.nbNoeuds): (params.nbU * params.nbNoeuds) + (params.nbX * (params.nbNoeuds + 1))]
P = sol[-params.nP:]

GRF = np.zeros((3, params.nbNoeuds))
q   = np.zeros((params.nbQ, params.nbNoeuds + 1))
dq  = np.zeros((params.nbQ, params.nbNoeuds + 1))
a   = np.zeros((params.nbMus, params.nbNoeuds))
f   = np.zeros((3, params.nbNoeuds))


# ------------ PHASE 1 : Stance phase
for k in range(params.nbNoeuds_stance):
    Uk = U[params.nbU*k : params.nbU*(k + 1)]
    Xk = X[params.nbX*k: params.nbX*(k + 1)]

    Q           = np.array(Xk[:params.nbQ]).squeeze()  # states
    dQ          = np.array(Xk[params.nbQ:2 * params.nbQ]).squeeze()
    activations = np.array(Uk[: params.nbMus]).squeeze()  # controls
    F           = np.array(Uk[params.nbMus :])

    q[:, k]  = Q
    dq[:, k] = dQ
    a[:, k]  = activations
    f[:, k]  = F

    X_int = int_RK4(ffcn_contact, params, Xk, Uk, P)
    q_int[:, k] = np.array(X_int[:params.nbQ]).squeeze()
    dq_int[:, k] = np.array(X_int[params.nbQ:2 * params.nbQ]).squeeze()
    constraint += X[params.nbX*(k + 1): params.nbX*(k + 2)] - X_int

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
    Uk = U[params.nbU * params.nbNoeuds_stance + params.nbU*k: params.nbU * params.nbNoeuds_stance + params.nbU*(k + 1)]
    Xk = X[params.nbX * params.nbNoeuds_stance + params.nbX * k: params.nbX * params.nbNoeuds_stance + params.nbX * (k + 1)]

    Q           = Xk[:params.nbQ]  # states
    dQ          = Xk[params.nbQ:2 * params.nbQ]
    activations = Uk[: params.nbMus]  # controls
    F           = Uk[params.nbMus :]

    q[:, params.nbNoeuds_stance + k] = Xk[:params.nbQ]
    dq[:, params.nbNoeuds_stance + k] = Xk[params.nbQ:]
    a[:, params.nbNoeuds_stance + k] = Uk[: params.nbMus]
    f[:, params.nbNoeuds_stance + k] = Uk[params.nbMus:]

    X_int = int_RK4(ffcn_no_contact, params, Xk, Uk, P)
    q_int[:, params.nbNoeuds_stance + k] = np.array(X_int[:params.nbQ]).squeeze()
    dq_int[:, params.nbNoeuds_stance + k] = np.array(X_int[params.nbQ:2 * params.nbQ]).squeeze()

    # OBJECTIVE FUNCTION
    Jm += fcn_objective_markers(params.wMa, params.wMt, Q, M_real_swing[:, :, k], 'swing')
    Je += fcn_objective_emg(params.wU, Uk, U_real_swing[:, k])
    Ja += fcn_objective_activation(params.wL, Uk)

q[:, -1]  = X[-2 * params.nbQ: -params.nbQ]
dq[:, -1] = X[-params.nbQ:]

while True:
    pass