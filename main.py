import numpy as np
import kmeans
import common
import naive_em
import em
import traceback


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

# K-Means run


def kmeans_run():
    X = np.loadtxt("toy_data.txt")
    K = [1, 2, 3, 4]
    seeds = [0, 1, 2, 3, 4]

    for k in K:
        for seed in seeds:
            (mixture, post) = common.init(X=X, K=k, seed=seed)
            (mixture, post, cost) = kmeans.run(X=X, mixture=mixture, post=post)

            plot_title = f"K-Means - K={k} Seed={seed} Cost={cost}"
            print(plot_title)
            common.plot(X=X, mixture=mixture, post=post,
                        title=plot_title)

# Naive EM Run


def naive_em_run():
    X = np.loadtxt("toy_data.txt")
    K = [1, 2, 3, 4]
    seeds = [0, 1, 2, 3, 4]

    for k in K:
        for seed in seeds:
            (mixture, post) = common.init(X=X, K=k, seed=seed)
            (mixture, post, cost) = naive_em.run(
                X=X, mixture=mixture, post=post)

            bic = common.bic(X, mixture, cost)
            plot_title = f"Naive EM - K={k} Seed={seed} Cost={cost} BIC={bic}"
            print(plot_title)
            # common.plot(X=X, mixture=mixture, post=post,
            #             title=plot_title)


def em_run():
    X = np.loadtxt("netflix_incomplete.txt")
    K = [1, 12]
    seeds = [0, 1, 2, 3, 4]

    all_costs = []
    for k in K:
        for seed in seeds:
            (mixture, post) = common.init(X, k, seed)
            (mixture, post, cost) = em.run(X, mixture, post)

            bic = common.bic(X, mixture, cost)
            plot_title = f"EM - K={k} Seed={seed} Cost={cost} BIC={bic}"
            print(plot_title)
            all_costs.append(cost)
            # common.plot(X, mixture, post, title=plot_title)

    best_cost = np.max(all_costs)
    print("Best Cost:", best_cost)
    return best_cost


def compare():
    X_gold = np.loadtxt('netflix_complete.txt')
    X = np.loadtxt("netflix_incomplete.txt")
    (mixture, post) = common.init(X, K=12, seed=1)
    (mixture, post, cost) = em.run(X, mixture, post)
    X_pred = em.fill_matrix(X, mixture)
    result = common.rmse(X_gold, X_pred)
    print("RMSE: ", result)


def main():
    try:
        # kmeans_run()
        # naive_em_run()
        # em_run()
        compare()
    except Exception:
        log_exit(traceback.format_exc())


if __name__ == "__main__":
    main()
