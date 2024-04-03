import torch

__all__ = ["fasta", "Convergence", "fastaOpts"]


class fastaOpts:
    def __init__(self, adaptive: bool = True, accelerate: bool = False, max_iters: int = 1000,
                 tol: float = 1e-3, window: int = 10):
        # The maximum number of iterations allowed before termination
        self.adaptive = adaptive
        self.accelerate = accelerate
        self.verbose = False
        self.max_iters = max_iters
        self.tol = tol
        self.recordIterates = False
        self.recordObjective = False
        self.stepSizeShrink = None
        self.window = window
        self.eps_n = 1e-8
        self.eps_r = 1e-8
        self.L = None
        self.tau = None
        self.stopNow = None
        self.backtrack = True
        self.restart = True
        self.stopRule = 'hybridResidual'


def setDefaultOpts(A, At, f, gradf, g, proxg, x0, opts: fastaOpts):
    if opts.stepSizeShrink is None:
        if opts.adaptive:
            # This is more aggressive, since the stepsize increases dynamically
            opts.stepSizeShrink = 0.2
        else:
            opts.stepSizeShrink = 0.5

    # Check if we need to approximate the Lipschitz constant of f
    if (opts.L is None or opts.L <= 0) and (opts.tau is None or opts.tau <= 0):
        # Compute two random vectors
        x1 = torch.randn_like(x0)
        x2 = torch.randn_like(x0)

        # Compute the gradients between the vectors
        gradf1 = At(gradf(A(x1)))
        gradf2 = At(gradf(A(x2)))

        # Approximate the Lipschitz constant of f
        opts.L = torch.linalg.norm((gradf1 - gradf2).ravel()) / torch.linalg.norm((x1 - x2).ravel())
        opts.L = max(opts.L, 1e-6)

        # We're guaranteed that FBS converges for tau <= 2.0 / L
        opts.tau = (2 / opts.L) / 10

    if (opts.tau is None) or (opts.tau <= 0):
        opts.tau = 1 / opts.L
    else:
        opts.L = 1 / opts.tau

    # stopping rule terminates when the relative residual gets small.
    if opts.stopNow is None:
        if opts.stopRule == 'hybridResidual':
            opts.stopNow = lambda x1, iter, resid, normResid, maxResidual, opts: resid / (
                    maxResidual + opts.eps_r) < opts.tol or normResid < opts.tol
        elif opts.stopRule == 'ratioResidual':
            opts.stopNow = lambda x1, iter, resid, normResid, maxResidual, opts: resid / (
                    maxResidual + opts.eps_r) < opts.tol
        elif opts.stopRule == 'normalizedResidual':
            opts.stopNow = opts.stopNow = lambda x1, iter, resid, normResid, maxResidual, opts: normResid < opts.tol
        elif opts.stopRule == 'residual':
            opts.stopNow = opts.stopNow = lambda x1, iter, resid, normResid, maxResidual, opts: resid < opts.tol
        elif opts.stopRule == 'iterations':
            opts.stopNow = lambda x1, iter, resid, normResid, maxResidual, opts: iter > opts.max_iters
        else:
            return False
    return True


def fasta(A, At, f, gradf, g, proxg, x0, opts: fastaOpts):
    # Option to just do gradient descent
    if g is None:
        g = lambda x: 0
        proxg = lambda x, t: x

    setDefaultOpts(A, At, f, gradf, g, proxg, x0, opts)

    # Record some frequently used information from opts
    tau1 = opts.tau  # initial stepsize
    max_iters = opts.max_iters  # maximum iterations before automatic termination
    w = opts.window  # lookback window for non-montone line search

    # Allocate memory
    residual = torch.zeros(max_iters)  # Residuals
    normalizedResid = torch.zeros(max_iters)  # Normalized residuals
    taus = torch.zeros(max_iters)  # Stepsizes
    fVals = torch.zeros(max_iters)  # The value of 'f', the smooth objective term
    objective = torch.zeros(max_iters + 1)  # The value of the objective function (f+g)
    totalBacktracks = 0  # How many times was backtracking activated

    # Initialize array values
    x1 = x0
    d1 = A(x1)
    f1 = f(d1)
    fVals[0] = f1
    gradf1 = At(gradf(d1))

    # Initialize additional storage required for FISTA
    if opts.accelerate:
        x_accel1 = x0;
        d_accel1 = d1;
        alpha1 = 1;

    if opts.recordIterates:
        iterates = torch.zeros((max_iters + 1, *x0.shape))
        iterates[0] = x1
    else:
        iterates = None

    # Handle non-monotonicity
    maxResidual = -torch.inf  # Stores the maximum value of the residual that has been seen. Used to evaluate stopping conditions.
    minObjectiveValue = torch.inf  # Stores the best objective value that has been seen. Used to return best iterate, rather than last iterate

    if opts.recordObjective:
        objective[0] = f1 + g(x0)

    # Begin Loop
    for i in range(max_iters):
        # Rename iterates relative to loop index. "0" denotes index i, and "1" denotes index i+1
        x0 = x1  # x_i <- x_{i+1}
        gradf0 = gradf1  # gradf0 is now ∇f(x_i)
        tau0 = tau1  # τ_i <- τ_{i+1}

        # FBS step: obtain x_{i+1} from x_i
        x1_hat = x0 - tau0 * gradf0  # Define x_{i+1}
        x1 = proxg(x1_hat, tau0)  # Define x_{i+1}

        # Non-monotone backtracking line search
        Dx = x1 - x0
        d1 = A(x1)
        f1 = f(d1)
        if opts.backtrack:
            M = torch.max(fVals[max(i - w, 0):max(i, 1)])  # Get largest of last w values of 'f'
            backtrackCount = 0
            # Note: 1e-12 is to quench rounding errors
            while (f1 - 1e-12 > M + Dx * gradf0 + torch.norm(Dx) ** 2 / (2 * tau0)) and (backtrackCount < 20):
                # The backtracking loop
                tau0 *= opts.stepSizeShrink  # shrink stepSize
                x1_hat = x0 - tau0 * gradf0  # redo the FBS
                x1 = proxg(x1_hat, tau0)
                d1 = A(x1)
                f1 = f(d1)
                Dx = x1 - x0
                backtrackCount += 1
            totalBacktracks += backtrackCount

        # Record information
        taus[i] = tau0  # stepSize
        residual[i] = torch.norm(Dx) / tau0  # Estimate of the gradient, should be zero at solution
        maxResidual = max(maxResidual, residual[i])
        normalizer = max(torch.norm(gradf0), torch.norm(x1 - x1_hat) / tau0) + opts.eps_n
        # Normalized residual: size of discrepancy between the two derivative terms, divided by the size of the terms
        normalizedResid[i] = residual[i] / normalizer
        fVals[i] = f1
        # record function values
        objective[i + 1] = f1 + g(x1)
        newObjectiveValue = objective[i + 1]

        if opts.recordIterates:  # record iterate values
            iterates[i + 1] = x1

        if newObjectiveValue < minObjectiveValue:  # Methods is non-monotone: Make sure to record best solution
            bestObjectiveIterate = x1
            minObjectiveValue = min(minObjectiveValue, newObjectiveValue)

        # If we stop, then record information in the output struct
        if opts.stopNow(x1, i + 1, residual[i], normalizedResid[i], maxResidual, opts):
            return Convergence(
                solution=bestObjectiveIterate,
                residuals=residual,
                objectives=objective,
                funcValues=fVals,
                iterates=iterates,
                stepsizes=taus,
                totalBacktracks=totalBacktracks,
                iterationCount=i + 1
            )

        if opts.accelerate:
            # FISTA-style acceleration, which works well for ill-conditioned problems
            # Rename last round's current variables to this round's previous variables
            x_accel0 = x_accel1
            d_accel0 = d_accel1

            x_accel1 = x1
            d_accel1 = d1

            alpha0 = alpha1

            # Prevent alpha from growing too large by restarting the acceleration
            if opts.restart and (x0 - x1).T @ (x1 - x_accel0) > 1E-30:
                alpha0 = 1.0

            # Recalculate acceleration parameter
            alpha1 = (1 + torch.sqrt(1 + 4 * alpha0 ** 2)) / 2

            # Overestimate the next value of x by a factor of (alpha0 - 1) / alpha
            # NOTE: this makes a copy of x1, which is necessary since x1's reference is linked to x0
            x1 = x_accel1 + (alpha0 - 1) / alpha1 * (x_accel1 - x_accel0)
            d1 = d_accel1 + (alpha0 - 1) / alpha1 * (d_accel1 - d_accel0)

            gradf1 = At(gradf(d1))
            fVals[i] = f(d1)
            tau1 = tau0

        elif opts.adaptive:
            # Compute stepsize needed for next iteration using BB/spectral method
            gradf1 = At(gradf(d1))
            # Delta_g, note that Delta_x was recorded above during backtracking
            Dg = gradf1 + (x1_hat - x0) / tau0
            dotprod = torch.real(Dx * Dg)
            tau_s = torch.norm(Dx) ** 2 / dotprod  # First BB stepsize rule
            tau_m = dotprod / torch.norm(Dg) ** 2  # Alternate BB stepsize rule
            tau_m = max(tau_m, 0)
            if 2 * tau_m > tau_s:  # Use "Adaptive" combination of tau_s and tau_m
                tau1 = tau_m
            else:
                tau1 = tau_s - 0.5 * tau_m  # Experiment with this param
            if tau1 <= 0 or torch.isinf(tau1) or torch.isnan(tau1):
                tau1 = tau0 * 1.5

        else:
            gradf1 = At(gradf(d1))
            tau1 = tau0

    return Convergence(
        solution=bestObjectiveIterate,
        residuals=residual,
        objectives=objective,
        funcValues=fVals,
        iterates=iterates,
        stepsizes=taus,
        totalBacktracks=totalBacktracks,
        iterationCount=max_iters
    )


class Convergence:
    def __init__(self, solution, objectives, funcValues, totalBacktracks, residuals, stepsizes, iterates,
                 iterationCount):
        """Record convergence information about FASTA.
        :param solution: The solution the algorithm computed
        :param residuals: The residuals, or the size differences between iterates, at each step
        :param iterates: The number of iterations until the algorithm converged
        :param stepsizes: The stepsizes at each step
        :param totalBacktracks: The number of backtracks performed at each step
        :param objectives: The value of the objective function at each step (default: None)
        :param funcValues: The value of f(x) at each step

        """
        self.solution = solution
        self.objectives = objectives
        self.funcVals = funcValues
        self.totalBacktracks = totalBacktracks
        self.residuals = residuals
        self.stepsizes = stepsizes
        self.iterates = iterates
        self.iterationCount = iterationCount
