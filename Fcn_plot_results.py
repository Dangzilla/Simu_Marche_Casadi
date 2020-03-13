from matplotlib import pyplot as plt
import numpy as np
import biorbd

def plot_q_MUSCOD(q0, T_phase, nbNoeuds_phase):
    # JOINT POSITIONS
    plt.figure(1)
    t_stance = np.linspace(0, T_phase[0], nbNoeuds_phase[0])
    t_swing = np.linspace(0, T_phase[1], nbNoeuds_phase[1])
    dt = t_swing[1]
    t_swing2 = t_stance[-1] + dt + t_swing
    t = np.hstack([t_stance, t_swing2])
    t = np.hstack([t, t[-1] + (t[-1] - t[-2])])

    plt.subplot(231)
    plt.title('Pelvis_Trans_X')
    plt.plot(t, q0[0, :], '+')
    plt.plot([T_phase[0], T_phase[0]], [min(q0[0, :]), max(q0[0, :])], 'k:')
    plt.xlabel('time (s)')
    plt.ylabel('position (m)')

    plt.subplot(232)
    plt.title('Pelvis_Trans_Y')
    plt.plot(t, q0[1, :], '+')
    plt.plot([T_phase[0], T_phase[0]], [min(q0[1, :]), max(q0[1, :])], 'k:')
    plt.xlabel('time (s)')
    plt.ylabel('position (m)')

    plt.subplot(233)
    plt.title('Pelvis_Rot_Z')
    plt.plot(t, q0[2, :]*180/np.pi, '+')
    plt.plot([T_phase[0], T_phase[0]], [min(q0[2, :]*180/np.pi), max(q0[2, :]*180/np.pi)], 'k:')
    plt.xlabel('time (s)')
    plt.ylabel('angle (deg)')

    plt.subplot(234)
    plt.title('R_Hip_Rot_Z')
    plt.plot(t, q0[3, :]*180/np.pi, '+')
    plt.plot([T_phase[0], T_phase[0]], [min(q0[3, :]*180/np.pi), max(q0[3, :]*180/np.pi)], 'k:')
    plt.xlabel('time (s)')
    plt.ylabel('angle (deg)')

    plt.subplot(235)
    plt.title('R_Knee_Rot_Z')
    plt.plot(t, q0[4, :]*180/np.pi, '+')
    plt.plot([T_phase[0], T_phase[0]], [min(q0[4, :]*180/np.pi), max(q0[4, :]*180/np.pi)], 'k:')
    plt.xlabel('time (s)')
    plt.ylabel('angle (deg)')

    plt.subplot(236)
    plt.title('R_Ankle_Rot_Z')
    plt.plot(t, q0[5, :]*180/np.pi, '+')
    plt.plot([T_phase[0], T_phase[0]], [min(q0[5, :]*180/np.pi), max(q0[5, :]*180/np.pi)], 'k:')
    plt.xlabel('time (s)')
    plt.ylabel('angle (deg)')

    plt.show(block=False)

def plot_dq_MUSCOD(dq0, T_phase, nbNoeuds_phase):
    # JOINT VELOCITIES
    plt.figure(2)
    t_stance = np.linspace(0, T_phase[0], nbNoeuds_phase[0])
    t_swing = t_stance[-1] + np.linspace(0, T_phase[1], nbNoeuds_phase[1])
    t = np.hstack([t_stance, t_swing])
    t = np.hstack([t, t[-1] + (t[-1] - t[-2])])

    plt.subplot(231)
    plt.title('Pelvis_Trans_X')
    plt.plot(t, dq0[0, :], '+-')
    plt.plot([T_phase[0], T_phase[0]], [min(dq0[0, :]), max(dq0[0, :])], 'k:')
    plt.xlabel('time (s)')
    plt.ylabel('speed (m/s)')

    plt.subplot(232)
    plt.title('Pelvis_Trans_Y')
    plt.plot(t, dq0[1, :], '+-')
    plt.plot([T_phase[0], T_phase[0]], [min(dq0[1, :]), max(dq0[1, :])], 'k:')
    plt.xlabel('time (s)')
    plt.ylabel('speed (m/s)')

    plt.subplot(233)
    plt.title('Pelvis_Rot_Z')
    plt.plot(t, dq0[2, :]*180/np.pi, '+-')
    plt.plot([T_phase[0], T_phase[0]], [min(dq0[2, :]*180/np.pi), max(dq0[2, :]*180/np.pi)], 'k:')
    plt.xlabel('time (s)')
    plt.ylabel('speed (deg/s)')

    plt.subplot(234)
    plt.title('R_Hip_Rot_Z')
    plt.plot(t, dq0[3, :]*180/np.pi, '+-')
    plt.plot([T_phase[0], T_phase[0]], [min(dq0[3, :]*180/np.pi), max(dq0[3, :]*180/np.pi)], 'k:')
    plt.xlabel('time (s)')
    plt.ylabel('speed (deg/s)')

    plt.subplot(235)
    plt.title('R_Knee_Rot_Z')
    plt.plot(t, dq0[4, :]*180/np.pi, '+-')
    plt.plot([T_phase[0], T_phase[0]], [min(dq0[4, :]*180/np.pi), max(dq0[4, :]*180/np.pi)], 'k:')
    plt.xlabel('time (s)')
    plt.ylabel('speed (deg/s)')

    plt.subplot(236)
    plt.title('R_Ankle_Rot_Z')
    plt.plot(t, dq0[5, :]*180/np.pi, '+-')
    plt.plot([T_phase[0], T_phase[0]], [min(dq0[5, :]*180/np.pi), max(dq0[5, :]*180/np.pi)], 'k:')
    plt.xlabel('time (s)')
    plt.ylabel('speed (deg/s)')

    plt.show(block=False)

def plot_markers_heatmap(diff_M):
    nbNoeuds = len(diff_M[2, 0, :])

    Labels_M = ["L_IAS", "L_IPS", "R_IPS", "R_IAS", "R_FTC",
                "R_Thigh_Top", "R_Thigh_Down", "R_Thigh_Front", "R_Thigh_Back", "R_FLE", "R_FME",
                "R_FAX", "R_TTC", "R_Shank_Top", "R_Shank_Down", "R_Shank_Front", "R_Shank_Tibia", "R_FAL", "R_TAM",
                "R_FCC", "R_FM1", "R_FMP1", "R_FM2", "R_FMP2", "R_FM5", "R_FMP5"]
    node = np.linspace(0, nbNoeuds, nbNoeuds, dtype=int)

    fig4, ax = plt.subplots()
    im = ax.imshow(diff_M[2, :, :])

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

def plot_emg_heatmap(diff_U):
    nbNoeuds   = len(diff_U[0, :])
    Labels_emg = ['GLUT_MAX1', 'GLUT_MED2', 'R_SEMITEN', 'R_BI_FEM_LH', 'R_RECTUS_FEM', 'R_VAS_MED', 'R_GAS_MED', 'R_GAS_LAT', 'R_SOLEUS', 'R_TIB_ANT']
    node       = np.linspace(0, nbNoeuds, nbNoeuds, dtype=int)

    fig, ax = plt.subplots()
    im_emg = ax.imshow(diff_U)

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


def plot_pelvis_force(F0, nbNoeuds_phase, T_phase):
    # Set time
    t_stance = np.linspace(0, T_phase[0], nbNoeuds_phase[0])
    t_swing = t_stance[-1] + np.linspace(0, T_phase[1], nbNoeuds_phase[1])
    t = np.hstack([t_stance, t_swing])

    plt.figure()
    plt.subplot(311)
    plt.title('Force Pelvis TX')
    plt.plot([0, t[-1]], [-1000, -1000], 'k--')  # lower bound
    plt.plot([0, t[-1]], [1000, 1000], 'k--')  # upper bound
    for n in range(nbNoeuds_phase[0] + nbNoeuds_phase[1] - 1):
        plt.plot([t[n], t[n + 1], t[n + 1]], [F0[0, n], F0[0, n], F0[0, n + 1]], 'b')

    plt.subplot(312)
    plt.title('Force Pelvis TY')
    plt.plot([0, t[-1]], [-2000, -2000], 'k--')  # lower bound
    plt.plot([0, t[-1]], [2000, 2000], 'k--')  # upper bound
    for n in range(nbNoeuds_phase[0] + nbNoeuds_phase[1] - 1):
        plt.plot([t[n], t[n + 1], t[n + 1]], [F0[1, n], F0[1, n], F0[1, n + 1]], 'b')

    plt.subplot(313)
    plt.title('Force Pelvis RZ')
    plt.plot([0, t[-1]], [-200, -200], 'k--')  # lower bound
    plt.plot([0, t[-1]], [200, 200], 'k--')  # upper bound
    for n in range(nbNoeuds_phase[0] + nbNoeuds_phase[1] - 1):
        plt.plot([t[n], t[n + 1], t[n + 1]], [F0[2, n], F0[2, n], F0[2, n + 1]], 'b')

    plt.show(block=False)


def plot_control_MUSCOD(u0, U_real, nbNoeuds_phase, T_phase):
    nbU   = len(u0[:, 0])
    nbMus = 17

    # plot control
    def plot_control(ax, t, x):
        nbPoints = len(np.array(x))
        for n in range(nbPoints - 2):
            ax.plot([t[n], t[n + 1], t[n + 1]], [x[n], x[n], x[n + 1]], 'b')

    # Set time
    t_stance = np.linspace(0, T_phase[0], nbNoeuds_phase[0])
    t_swing = t_stance[-1] + np.linspace(0, T_phase[1], nbNoeuds_phase[1])
    t = np.hstack([t_stance, t_swing])

    # CONTROL
    fig1, axes1 = plt.subplots(5, 4, sharex=True, figsize=(10, 10))
    Labels = ['GLUT_MAX1', 'GLUT_MAX2', 'GLUT_MAX3', 'GLUT_MED1', 'GLUT_MED2', 'GLUT_MED3',
              'R_SEMIMEM', 'R_SEMITEN', 'R_BI_FEM_LH', 'R_RECTUS_FEM', 'R_VAS_MED', 'R_VAS_INT',
              'R_VAS_LAT', 'R_GAS_MED', 'R_GAS_LAT', 'R_SOLEUS', 'R_TIB_ANT', 'Pelvis Tx', 'Pelvis Ty', 'Pelvis Rz']    # Control labels
    axes1 = axes1.flatten()
    u_emg = 9
    for i in range(nbU):
        ax = axes1[i]  # get the correct subplot
        ax.set_title(Labels[i])  # put control label
        ax.plot([T_phase[0], T_phase[0]], [0, 1], 'k:')  # end of the stance phase
        plot_control(ax, t, u0[i, :])
        ax.grid(True)
        if (i != 1) and (i != 2) and (i != 3) and (i != 5) and (i != 6) and (i != 11) and (i != 12) and (i < nbMus):
            ax.plot(t, U_real[u_emg, :], 'r')  # plot emg if available
            u_emg -= 1
        if (i > nbU - 5):
            ax.set_xlabel('time (s)')
        if (i < (nbMus - 1)):
            ax.plot([0, t[-1]], [0, 0], 'k--')  # lower bound
            ax.plot([0, t[-1]], [1, 1], 'k--')  # upper bound
            ax.yaxis.set_ticks(np.arange(0, 1.5, 0.5))
        else:
            f_lb = [-1000, -2000, -200]
            f_ub = [1000, 2000, 200]

            for f in range(3):
                ax = axes1[nbMus + f]
                ax.plot([0, t[-1]], [f_lb[f], f_lb[f]], 'k--')  # lower bound
                ax.plot([0, t[-1]], [f_ub[f], f_ub[f]], 'k--')  # upper bound
    plt.show(block=False)


def plot_GRF_MUSCOD(GRF, GRF_real, nbNoeuds_phase, T_phase):
    # Set time
    t_stance = np.linspace(0, T_phase[0], nbNoeuds_phase[0])
    t_swing = t_stance[-1] + np.linspace(0, T_phase[1], nbNoeuds_phase[1])
    t = np.hstack([t_stance, t_swing])

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes.flatten()

    ax_ap = axes[0]
    ax_ap.set_title('GRF A/P  during the gait')
    ax_ap.plot(t, GRF_real[1, :], 'r')
    ax_ap.plot([T_phase[0], T_phase[0]], [min(GRF_real[1, :]), max(GRF_real[1, :])], 'k:')  # end of the stance phase
    ax_ap.plot(t, GRF[0, :], 'b')
    ax_ap.grid(True)

    ax_v = axes[1]
    ax_v.set_title('GRF vertical')
    ax_v.plot(t, GRF_real[2, :], 'r')
    ax_v.plot([T_phase[0], T_phase[0]], [min(GRF_real[2, :]), max(GRF_real[2, :])], 'k:')
    ax_v.plot(t, GRF[2, :], 'b')
    ax_v.set_xlabel('time (s)')
    ax_v.grid(True)
    fig.tight_layout()
    plt.show(block=False)

def plot_markers_result(sol_q, T_phase, nbNoeuds_phase, nbMarker, M_real):
    # Plot leg trajectory with 5 markers

    # INPUT
    # sol_q          = optimized joint position (nbQ x nbNoeuds)
    # nbNoeuds_phase = shooting points for each phase (nbPhase)
    # nbMarker       = number of markers

    # PARAMETERS
    nbNoeuds = len(sol_q[0, :]) - 1
    nbPhase  = len(T_phase)
    nbQ      = len(sol_q[:, 0])
    M_simu   = np.zeros((3, nbMarker, nbNoeuds + 1))

    plt.figure()
    # FIND MARKERS POSITIONS
    for k_stance in range(nbNoeuds_phase[0]):
        model   = biorbd.Model('/home/leasanchez/programmation/Marche_Florent/ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod')
        markers = model.markers(sol_q[:, k_stance])
        for nMark in range(nbMarker):
            M_simu[:, nMark, k_stance] = markers[nMark].to_array()

        # PLOT ARTIFICIAL SEGMENTS TO FOLLOW LEG MOVEMENT
        M_aff = np.zeros((3, 5))
        M_aff[:, 0] = M_simu[:, 2, k_stance]
        M_aff[:, 1] = M_simu[:, 4, k_stance]
        M_aff[:, 2] = M_simu[:, 11, k_stance]
        M_aff[:, 3] = M_simu[:, 19, k_stance]
        M_aff[:, 4] = M_simu[:, 22, k_stance]
        plt.plot(M_aff[0, :], M_aff[2, :], 'bo-', alpha=0.5)
        plt.plot([M_real[0, 2, k_stance], M_real[0, 4, k_stance], M_real[0, 11, k_stance], M_real[0, 19, k_stance], M_real[0, 22, k_stance]],
                 [M_real[2, 2, k_stance], M_real[2, 4, k_stance], M_real[2, 11, k_stance], M_real[2, 19, k_stance], M_real[2, 22, k_stance]], 'r+')

    for k_swing in range(nbNoeuds_phase[1] + 1):
        model = biorbd.Model('/home/leasanchez/programmation/Marche_Florent/ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod')
        markers = model.markers(sol_q[:, nbNoeuds_phase[0] + k_swing])
        for nMark in range(nbMarker):
            M_simu[:, nMark, nbNoeuds_phase[0] + k_swing] = markers[nMark].to_array()

        # PLOT ARTIFICIAL SEGMENTS TO FOLLOW LEG MOVEMENT
        M_aff = np.zeros((3, 5))
        M_aff[:, 0] = M_simu[:, 2, nbNoeuds_phase[0] + k_swing]
        M_aff[:, 1] = M_simu[:, 4, nbNoeuds_phase[0] + k_swing]
        M_aff[:, 2] = M_simu[:, 11, nbNoeuds_phase[0] + k_swing]
        M_aff[:, 3] = M_simu[:, 19, nbNoeuds_phase[0] + k_swing]
        M_aff[:, 4] = M_simu[:, 22, nbNoeuds_phase[0] + k_swing]
        plt.plot(M_aff[0, :], M_aff[2, :], 'go-', alpha=0.5)
        plt.plot([M_real[0, 2, nbNoeuds_phase[0] + k_swing], M_real[0, 4, nbNoeuds_phase[0] + k_swing], M_real[0, 11, nbNoeuds_phase[0] + k_swing], M_real[0, 19, nbNoeuds_phase[0] + k_swing], M_real[0, 22, nbNoeuds_phase[0] + k_swing]],
                 [M_real[2, 2, nbNoeuds_phase[0] + k_swing], M_real[2, 4, nbNoeuds_phase[0] + k_swing], M_real[2, 11, nbNoeuds_phase[0] + k_swing], M_real[2, 19, nbNoeuds_phase[0] + k_swing], M_real[2, 22, nbNoeuds_phase[0] + k_swing]], 'm+')

    plt.plot([-0.5, 1.5], [0, 0], 'k--')

    plt.show(block = False)

