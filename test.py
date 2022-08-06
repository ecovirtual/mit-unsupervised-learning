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


def main():
    try:
        test_estep()
    except Exception:
        log_exit(traceback.format_exc())


if __name__ == "__main__":
    main()
