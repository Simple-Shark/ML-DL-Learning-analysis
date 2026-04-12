import numpy as np

"""
    k-means聚类 
    在一个多维向量里面 每个 样本的特征值作为每个点的具体坐标系, 如果用 欧式距离进行聚类 需要注意内部实现的是向量加法
    并非整数加法 因为每个样本可能存在多维特征,需要都进行考虑

"""


class k_means():
    def __init__(self, n_clusters, max_epoch=400, init="k-means++", vary=1e-6):
        self.n_clusters = n_clusters
        self.init = init
        self.max_epoch = max_epoch
        self.centers = None
        self.labels = None
        self.vary = vary

    def _compute_distances(self, x, centers):
        """
            计算距离函数 首先得获取长度   为每个簇心进行计算
               distant就可以构建成一个[,len(centers]]这样每个簇心都能算到
               于此同时 也得为每个样本保存相应的distant 所以 distant需构建成[x.shape[0],]的形式
               根据以下算法即可为每个簇心计算距离了
        :param x:
        :param centers:
        :return distant:
        """
        distant = np.zeros((x.shape[0], len(centers)))
        for i, center in enumerate(centers):
            distant[:, i] = np.sqrt(np.sum((x - center) ** 2, axis=1))
        return distant

    def _initialization(self, x):
        """
            初始化参数 主要是根据 self.n_clusters来随机获取簇心
        :param x:
        :return centers[n_clusters,num_feature]
        """

        num_sample, num_feature = x.shape
        if self.init == "k-means++":
            centers = []

            # 步骤 1: 随机选择第一个中心
            first_idx = np.random.randint(0, num_sample)
            centers.append(x[first_idx])
            for _ in range(1, self.n_clusters):
                # 计算每个样本到最近中心的距离平方
                dist_matrix = self._compute_distances(x, centers)
                min_distances = np.min(dist_matrix, axis=1)
                distances_squared = min_distances ** 2
                probabilities = distances_squared / np.sum(distances_squared)  # 计算选择概率

                cumulative_probs = np.cumsum(probabilities)  # 轮盘赌选择下一个中心
                random_val = np.random.random()

                next_idx = np.searchsorted(cumulative_probs, random_val)
                next_idx = np.clip(next_idx, 0, num_sample - 1)

                centers.append(x[next_idx])
            return centers

    def fit(self, x):
        """                 ——————————模型训练————————————
                        算法原理: 基于初始化随机获得的 簇心组
                        让每个样本与 簇心作差取最小值 与最小值的簇归为一类
                        完成一轮遍历后 取每个簇心组的均值，重新执行以上操作
                        直到变化率很小为止，这里可以使用肘部法获取最佳k值
        :param x:
        :return:
        """
        centers = self._initialization(x)  # 获取簇心数组 需要不断更新
        pre_centers = None
        """

                样本归类算法 : 先用距离函数计算 每个样本到簇心的距离  labels对每个样本进行argmin取得最小值归为一种簇 
                            用布尔索引x[label==i]进行分类 计算完成后 , 开始更新簇心

                簇心更新算法: 在上述算法完成的基础上 进行循环 取每个簇样本特征列的平均值作为 在多维空间中 新簇心的坐标
                            后再次计算 是否有新样本变化  变化小就直接跳出循环 ,有变化就反复进行上述算法运算  直到达到max_epoch 
                             跳出循环返回self 得到最优参数  

        """
        for _ in range(self.max_epoch):
            distance = self._compute_distances(x, centers)
            labels = np.argmin(distance, axis=1)
            new_centers = []

            for i in range(self.n_clusters):
                cluster_sample = x[labels == i]
                if len(cluster_sample) > 0:
                    new_center = np.mean(cluster_sample, axis=0)
                else:
                    new_center = centers[i]
                new_centers.append(new_center)

            if pre_centers is not None:
                if np.sum([np.linalg.norm(new - old) for new, old in zip(new_centers, pre_centers)]) < self.vary:
                    break

            pre_centers = centers.copy() if isinstance(centers, list) else [c.copy() for c in centers]
            centers = new_centers
            self.centers = np.array(centers)
            self.labels = labels
        return self

    def predict(self, x):
        distance = self._compute_distances(x, centers=self.centers)
        return np.argmin(distance, axis=1)





