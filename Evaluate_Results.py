import Fcn_Objective
import LoadData

def calculate_objectivefcn(params, X, U):
    Ja = 0  # objective function for muscle activation
    Jm = 0  # objective function for markers
    Je = 0  # objective function for EMG
    JR = 0  # objective function for ground reactions

    # ----------------------------- Load Data from c3d file ------------------------------------------------------------
    # GROUND REACTION FORCES & SET TIME
    [GRF_real, params.T, params.T_stance, params.T_swing] = LoadData.load_data_GRF(params, 'cycle')

    # MARKERS POSITION
    M_real_stance = LoadData.load_data_markers(params, 'stance')
    M_real_swing  = LoadData.load_data_markers(params, 'swing')

    # MUSCULAR EXCITATION
    U_real_swing  = LoadData.load_data_emg(params, 'swing')
    U_real_stance = LoadData.load_data_emg(params, 'stance')

    # ----------------------------- Rearrange data ---------------------------------------------------------------------
    q  = [X[0::params.nbX], X[1::params.nbX], X[2::params.nbX], X[3::params.nbX], X[4::params.nbX], X[5::params.nbX]]
    dq = [X[6::params.nbX], X[7::params.nbX], X[8::params.nbX], X[9::params.nbX], X[10::params.nbX], X[11::params.nbX]]
    a  = [U[0::params.nbU], U[1::params.nbU], U[2::params.nbU], U[3::params.nbU], U[4::params.nbU], U[5::params.nbU], U[6::params.nbU],
          U[7::params.nbU], U[8::params.nbU], U[9::params.nbU], U[10::params.nbU], U[11::params.nbU], U[12::params.nbU], U[13::params.nbU],
          U[14::params.nbU], U[15::params.nbU], U[16::params.nbU]]
    F  = [U[17::params.nbU], U[18::params.nbU], U[19::params.nbU]]


    # ------------ PHASE 1 : Stance phase
    for k in range(params.nbNoeuds_stance):
        # DYNAMIQUE
        Uk = U[params.nbU*k: params.nbU*(k + 1)]
        Xk = X[params.nbX*k: params.nbX*(k + 1)]

        # OBJECTIVE FUNCTION
        [grf, Jr] = Fcn_Objective.fcn_objective_GRF(params.wR, Xk, Uk, GRF_real[:, k])                                                    # tracking ground reaction --> stance
        JR += Jr
        Jm += Fcn_Objective.fcn_objective_markers(params.wMa, params.wMt, Xk[: params.nbQ], M_real_stance[:, :, k], 'stance')             # tracking marker
        Je += Fcn_Objective.fcn_objective_emg(params.wU, Uk, U_real_stance[:, k])                                                         # tracking emg
        Ja += Fcn_Objective.fcn_objective_activation(params.wL, Uk)                                                                       # min muscle activations (no EMG)

    # ------------ PHASE 2 : Swing phase
    for k in range(params.nbNoeuds_swing):
        # DYNAMIQUE
        Uk = U[params.nbU * params.nbNoeuds_stance + params.nbU*k: params.nbU * params.nbNoeuds_stance + params.nbU*(k + 1)]
        Xk = X[params.nbX * params.nbNoeuds_stance + params.nbX*k: params.nbX * params.nbNoeuds_stance + params.nbX*(k + 1)]

        # OBJECTIVE FUNCTION
        Jm += Fcn_Objective.fcn_objective_markers(params.wMa, params.wMt, Xk[: params.nbQ], M_real_swing[:, :, k], 'swing')               # tracking marker
        Je += Fcn_Objective.fcn_objective_emg(params.wU, Uk, U_real_swing[:, k])                                                          # tracking emg
        Ja += Fcn_Objective.fcn_objective_activation(params.wL, Uk)

    return [Ja, Je, Jm, JR]
