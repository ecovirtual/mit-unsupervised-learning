import numpy as np
import em
import common
import traceback
import naive_em

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 4
n, d = X.shape
seed = 0

# TODO: Your code here


def green(s):
    return '\033[1;32m%s\033[m' % s


def yellow(s):
    return '\033[1;33m%s\033[m' % s


def red(s):
    return '\033[1;31m%s\033[m' % s


def log(*m):
    print(" ".join(map(str, m)))


def log_exit(*m):
    log(red("ERROR:"), *m)
    exit(1)


def test_estep():
    X = np.loadtxt("toy_data.txt")
    K = 3
    seed = 0

    (mixture, post) = common.init(X=X, K=K, seed=seed)

    (counts, log_likelihood) = naive_em.estep(X, mixture)
    print(F"Log Likelihood: {log_likelihood}")

    # mixture = naive_em.mstep(X, post)
    # print(F"Mixture:  {mixture}")

    correct_ll = -1388.0818

    if (round(log_likelihood, 4) == correct_ll):
        log(green("PASS"), "E-Step")
    else:
        log(red("FAILED"), "E-Step")


def test_mstep():
    X = np.array([[0.85794562, 0.84725174],
                  [0.6235637, 0.38438171],
                  [0.29753461, 0.05671298],
                  [0.27265629, 0.47766512],
                  [0.81216873, 0.47997717],
                  [0.3927848, 0.83607876],
                  [0.33739616, 0.64817187],
                  [0.36824154, 0.95715516],
                  [0.14035078, 0.87008726],
                  [0.47360805, 0.80091075],
                  [0.52047748, 0.67887953],
                  [0.72063265, 0.58201979],
                  [0.53737323, 0.75861562],
                  [0.10590761, 0.47360042],
                  [0.18633234, 0.73691818]])
    K = 6

    (mixture, post) = common.init(X=X, K=K)

    mixture = naive_em.mstep(X, post)

    # correct_mu = np.array([
    #     [0.43216722, 0.64675402],
    #     [0.46139681, 0.57129172],
    #     [0.44658753, 0.68978041],
    #     [0.44913747, 0.66937822],
    #     [0.47080526, 0.68008664],
    #     [0.40532311, 0.57364425]])
    # correct_var = np.array(
    #     [0.05218451, 0.06230449, 0.03538519, 0.05174859, 0.04524244, 0.05831186])
    # correct_p = np.array([0.1680912, 0.15835331, 0.21384187,
    #                      0.14223565, 0.14295074, 0.17452722])

    if ((mixture.mu == correct_mu).all()):
        log(green("PASS"), "M-Step")
    else:
        log(red("FAILED"), "M-Step")


def test_naive_em():
    X = np.loadtxt("toy_data.txt")
    K = 6
    seed = 0

    (mixture, post) = common.init(X=X, K=K, seed=seed)
    (mixture, post, new_cost) = naive_em.run(X, mixture, post)

    # print(F"Mixture: {mixture}")
    # print(F"Post: {post}")
    print(F"New Cost: {new_cost}")


def test_bci():
    X = np.array([[0.85794562, 0.84725174],
                  [0.6235637, 0.38438171],
                  [0.29753461, 0.05671298],
                  [0.27265629, 0.47766512],
                  [0.81216873, 0.47997717],
                  [0.3927848, 0.83607876],
                  [0.33739616, 0.64817187],
                  [0.36824154, 0.95715516],
                  [0.14035078, 0.87008726],
                  [0.47360805, 0.80091075],
                  [0.52047748, 0.67887953],
                  [0.72063265, 0.58201979],
                  [0.53737323, 0.75861562],
                  [0.10590761, 0.47360042],
                  [0.18633234, 0.73691818]])
    mu = np.array([[0.6235637, 0.38438171],
                   [0.3927848, 0.83607876],
                   [0.81216873, 0.47997717],
                   [0.14035078, 0.87008726],
                   [0.36824154, 0.95715516],
                   [0.10590761, 0.47360042]])
    var = np.array([0.10038354, 0.07227467, 0.13240693,
                    0.12411825, 0.10497521, 0.12220856])
    p = np.array([0.1680912, 0.15835331, 0.21384187,
                 0.14223565, 0.14295074, 0.17452722])
    ll = -1065.772989
    correct_bci = -1096.915566

    bci = common.bic(X, common.GaussianMixture(mu, var, p), log_likelihood=ll)

    print("BCI: ", bci)

    if (round(bci, 6) == correct_bci):
        log(green("PASS"), "BCI")
    else:
        log(red("FAILED"), "BCI")


def main():
    try:
        # test_estep()
        # test_mstep()
        # test_naive_em()
        test_bci()
    except Exception:
        log_exit(traceback.format_exc())


if __name__ == "__main__":
    main()
