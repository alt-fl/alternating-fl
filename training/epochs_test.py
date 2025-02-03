from epochs import *

if __name__ == "__main__":
    # code for testing the different transition functions
    n = 10
    budget = 3000

    def disp(trans):
        return [trans.estimate_epoch(round) for round in range(0, n)]

    lin_est_max = estimate_max_epoch(budget, LinearTransition, n=n)
    print("Linear estimated max epoch", lin_est_max)
    lin = LinearTransition(lin_est_max, n=n)
    print("Linear budget spent", get_total_epochs(lin, n=n))
    print("Linear", disp(lin))

    exp_est_max = estimate_max_epoch(budget, ExponentialTransition, n=n)
    print("Exp estimated max epoch", exp_est_max)
    exp = ExponentialTransition(exp_est_max, n=n)
    print("Exp budget spent", get_total_epochs(exp, n=n))
    print("Exp", disp(exp))

    inv_var_est_max = estimate_max_epoch(budget, InverseVariationTransition, n=n)
    print("Inverse variation estimated max epoch", inv_var_est_max)
    inv_var = InverseVariationTransition(inv_var_est_max, n=n)
    print("Inverse variation budget spent", get_total_epochs(inv_var, n=n))
    print("Inverse variation", disp(inv_var))

    log_est_max = estimate_max_epoch(budget, LogarithmicTransition, n=n)
    print("Log estimated max epoch", log_est_max)
    log = LogarithmicTransition(log_est_max, n=n)
    print("Log budget spent", get_total_epochs(log, n=n))
    print("Log", disp(log))

    quad_est_max = estimate_max_epoch(budget, QuadraticTransition, n=n)
    print("Quad estimated max epoch", quad_est_max)
    quad = QuadraticTransition(quad_est_max, n=n)
    print("Quad budget spent", get_total_epochs(quad, n=n))
    print("Quad", disp(quad))
