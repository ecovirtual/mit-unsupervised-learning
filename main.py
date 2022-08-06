import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

#
K = 4
seed = 4

(mixture, post) = common.init(X=X, K=K, seed=seed)
(mixture, post, cost) = kmeans.run(X=X, mixture=mixture, post=post)

plot_title = f"K={K} Seed={seed} Cost={cost}"
print(plot_title)
common.plot(X=X, mixture=mixture, post=post,
            title=plot_title)
