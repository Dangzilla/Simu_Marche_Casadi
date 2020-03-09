import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def InitialGuess_MUSCOD(file, nbQ, nbMus, nP, nbNoeuds_phase):
    nbX   = 2*nbQ + nbMus
    nbU   = nbMus + 3

    f = open(file, 'r')
    content = f.read()
    content_divide = content.split('\n')

    # FIND STATE -- sd
    x0 = np.zeros((nbX, (nbNoeuds_phase[0] + nbNoeuds_phase[1] + 1)))
    # phase 1 -- stance
    for n in range(nbNoeuds_phase[0]):
        state = 'sd(0,' + str(n) + ')  ! ' + str(n)
        idx = content_divide.index(state)
        for x in range(nbX):
            a = content_divide[idx + x + 1].split(':')
            x0[x, n] = float(a[1])
    # phase 2 -- swing
    for n in range(nbNoeuds_phase[1]):
        state = 'sd(1,' + str(n) + ')  ! ' + str(nbNoeuds_phase[0] + n)
        idx = content_divide.index(state)
        for x in range(nbX):
            a = content_divide[idx + x + 1].split(':')
            x0[x, (nbNoeuds_phase[0] + n)] = float(a[1])

    # phase ~3 -- impact
    state = 'sd(2,0)  ! 50'
    idx = content_divide.index(state)
    for x in range(nbX):
        a = content_divide[idx + x + 1].split(':')
        x0[x, (nbNoeuds_phase[0] + nbNoeuds_phase[1])] = float(a[1])


    # FIND CONTROL -- u
    u0 = np.zeros((nbU, (nbNoeuds_phase[0] + nbNoeuds_phase[1])))
    for n in range(nbNoeuds_phase[0]):
        control = 'u(0,' + str(n) + ')  ! ' + str(n)
        idx = content_divide.index(control)
        for u in range(nbU):
            a = content_divide[idx + u + 1].split(':')
            u0[u, n] = float(a[1])
    # phase 2 -- swing
    for n in range(nbNoeuds_phase[1]):
        control = 'u(1,' + str(n) + ')  ! ' + str(nbNoeuds_phase[0] + n)
        idx = content_divide.index(control)
        for u in range(nbU):
            a = content_divide[idx + u + 1].split(':')
            u0[u, (nbNoeuds_phase[0] + n)] = float(a[1])


    # FIND PARAMETERS FOR ISOMETRIC FORCE
    p0    = np.zeros(nP)
    param = 'p'
    idx   = content_divide.index(param)
    for p in range(nP):
        a = content_divide[idx + p + 1].split(':')
        p0[p] = float(a[1])

    return u0, x0, p0

