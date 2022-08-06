import numpy as np
import kmeans
import common
import naive_em
import em
import traceback


def log(*m):
    print(" ".join(map(str, m)))


def log_exit(*m):
    log(red("ERROR:"), *m)
    exit(1)


X = np.loadtxt("toy_data.txt")

# K-Means run


def kmeans_run():
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
    K = [1, 2, 3, 4]
    seeds = [0, 1, 2, 3, 4]

    for k in K:
        for seed in seeds:
            (mixture, post) = common.init(X=X, K=k, seed=seed)
            (mixture, post, cost) = naive_em.run(
                X=X, mixture=mixture, post=post)

            plot_title = f"Naive EM - K={k} Seed={seed} Cost={cost}"
            print(plot_title)
            common.plot(X=X, mixture=mixture, post=post,
                        title=plot_title)


def main():
    try:
        kmeans_run()
        # naive_em()
    except Exception:
        log_exit(traceback.format_exc())


if __name__ == "__main__":
    main()
