import numpy as np
import pandas as pd
class Linear():  # logitic classification  Tow
    def __init__(self, learn_rate=0.01):
        self.w = None
        self.b = None
        self.learn_rate = learn_rate
        self.out_label = 0

    def __init_params(self, x, label):
        if self.out_label == 0:
            self.out_label = np.unique(label).shape[0]
        if self.w is None:
            limit = np.sqrt(6 / (x.shape[1] + 1))
            self.w = np.random.uniform(-limit, limit, (x.shape[1],))
            self.b = 0.0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, x, label):
        self.__init_params(x, label)
        z = x @ self.w + self.b
        return np.sum(-label * z + np.log(1 + np.exp(z)))

    def matrix(self, x):
        x_b = np.hstack([x, np.ones((x.shape[0], 1))])
        beta = np.hstack([self.w, [self.b]])
        return x_b, beta

    def forward(self, x_b, beta):
        z = x_b @ beta
        return self.sigmoid(z)

    def gradient(self, x, label):
        """
         梯度公式：-x_b.T @ (y - y_pred) / n
        """
        try:
            x_b, beta = self.matrix(x)
            y_pre = self.forward(x_b, beta)
            error = label - y_pre
            n = len(label)
            grad = -x_b.T @ error / n  # (n_features+1,)‘

            return grad

        except Exception as e:
            print(f"wrong！！！{e}")
        return None

    def predict(self, x):
        x_b, beta = self.matrix(x)
        return self.forward(x_b, beta)

    def fit(self, x, label, epochs=1000):
        """
         :param x:
         :param label:
         :param epochs:      用于更新参数训练模型
         :return:
        """
        self.__init_params(x, label)
        for epoch in range(epochs):
            grad = self.gradient(x, label)
            beta = np.hstack([self.w, self.b])
            beta -= self.learn_rate * grad  # 同时更新 w 和 b
            self.w = beta[:-1]  # 分离回去
            self.b = beta[-1]
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {self.loss(x, label):.4f}")
