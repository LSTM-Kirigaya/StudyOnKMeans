import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

plt.rc('font', family='Times New Roman')    # 字体
scatter_dict = {        # 散点的属性
    "s" : 80,       # 散点半径
    "alpha" : 0.75      # 散点透明度
}

cluster_dict = {    # 聚类中心的属性
    "s" : 150,
    "marker" : "*"
}

np.random.seed(100)     # 随机数种子

center = 0.03
radius = 0.07

points1 = np.random.uniform(0, radius, [25, 2]) + center    # 以(center, center)为中心的25个随机点
points2 = np.random.uniform(0, radius, [25, 2]) - center    # 以(-center, -center)为中心的25个随机点
all_points = np.concatenate([points1, points2], axis=0)     # 合并上面这50个随机点

init_cluster1 = np.array([[0.03, 0.1], [0.1, 0.03]])    # 初始选取的两个聚类中心

cluster_model = KMeans(n_clusters=2, init=init_cluster1, n_init=1)
cluster_model.fit(X=all_points)

plt.scatter(all_points[:, 0], all_points[:, 1], c=cluster_model.labels_, **scatter_dict)    # 根据划分的标签画出散点图
plt.scatter(cluster_model.cluster_centers_[:, 0], cluster_model.cluster_centers_[:, 1], **cluster_dict)    # 绘制聚类中心
plt.show()
