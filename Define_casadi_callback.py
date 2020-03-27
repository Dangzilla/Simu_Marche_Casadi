from casadi import *

class AnimateCallback(Callback):
    def __init__(self, name, nx, ng, np, opts={}):
        Callback.__init__(self)
        self.nx     = nx                   # optimized value number
        self.ng     = ng                   # constraints number
        self.nP     = np

        self.sol_data   = None             # optimized variables
        self.obj_value  = None             # objective fcn value
        self.constaint  = None             # constraint values
        self.update_sol = False            # first iteration

        self.construct(name, opts)

    def get_n_in(self): return nlpsol_n_out()
    def get_n_out(self): return 1
    def get_name_in(self, i): return nlpsol_out(i)
    def get_name_out(self, i): return "ret"

    def get_sparsity_in(self, i):
        n = nlpsol_out(i)
        if n == 'f': return Sparsity.scalar()
        elif n in ('x', 'lam_x'): return Sparsity.dense(self.nx)
        elif n in ('g', 'lam_g'): return Sparsity.dense(self.ng)
        else: return Sparsity(0, 0)

    def eval(self, arg):
        darg = {}
        for (i, s) in enumerate(nlpsol_out()): darg[s] = arg[i]

        self.sol_data   = darg['x']
        self.obj_value  = darg['f']
        self.constaint  = darg['g']
        self.update_sol = True
        return [0]