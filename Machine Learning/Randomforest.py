import numpy as np
import pandas as pd
from Decistion_Tree import *
"""        _________________随机森林算法实现________________________
        1.基于上次所书写的cart决策树
        2.随机森林是一个多学习器的算法
        3.基于多个学习器得出的结果进行投票选择得出最终结果

"""
class RandomForest:

    def __init__(self, n_estimators, max_depth,data_sampling="replacement",same_rate=0.5):
        """

        :param n_estimators:学习器的个数
        :param max_depth: 最大深度
        :param data_sampling:数据采样方式
        :param same_rate:学习器之间相同样本的相同率

        样本n1 样本n2    same_rate= D(n1 ∩ n2)/n1+n2
        每个学习器不同的个数需要达到  len(x)//self.n_estimators

        为防止模型对某一类别过拟合 所以对每个类别样本都进行抽取
        在每个样本取数都是取一半
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
        self.trees_data=[]
        self.data_sampling = data_sampling
        self.same_rate = same_rate if data_sampling=="replacement" and same_rate>0 and same_rate<1 else None

    def get_unique_label(self,x):   #获取单列特征或样本的 类别,及对应的个数
        values, count = np.unique(x, return_counts=True)
        dic = dict(zip(values, count))
        return values,dic


    def __index_generator(self,y,seed):#利用内置random函数进行随机取数 保证每个类别都有样本被取到
        label_list, dic = self.get_unique_label(y)
        index_list = []
        label_index_list = []
        rng = np.random.default_rng(seed)
        for index, value in enumerate(label_list):
            label_index_list.append(np.where(y == value)[0])
        if self.data_sampling == "replacement":
            for i, values in enumerate(label_index_list):

                size = max(1, int(dic[label_list[i]] * self.same_rate))
                index_list.append(rng.choice(values, size=size))
        else:
            for i ,values in enumerate(label_index_list):
                size=max(1,int(dic[label_list[i]]//self.n_estimators))
                for j in range(size):
                    index_list.append(values[size*j:size*(j+1)])   #获取一个不放回随机抽样的样本索引
        return index_list

    def sampling_with_replacement(self,x,y):#实现不同学习器进行放回抽样
        """
            根据学习器的个数对数据进行遍历
            要保证每个学习器的样本相同
        :param x:
        :param y:
        :return:
        """
        for i in range(self.n_estimators):
            #获取索引数组
            index=self.__index_generator(y,i)
            index = np.concatenate(index)
            index = np.unique(index)
            Feature=x[index,:]
            label=y[index]
            self.trees_data.append([Feature,label])
        return self



    def sampling_without_replacement(self,x,y):#实现不同学习器的不放回抽样
        #对每个样本按顺序抽样
        """
            对每个类别样本每个学习器完成
            (n_sample//self.n_estimators)个的抽取
        :param x:
        :param y:
        :return:
        """

        n_list=[]
        for i in range(self.n_estimators):
            pass

    def fit(self, x, y):
        """
            模型训练 实质训练n_estimators 弱训练器,如果是有放回抽样的就可以进行投票获取最终结果

        :param x:Feature 特征
        :param y: label 类别或者目标值
        :return: 返回self.tree
        """
        self.trees = []
        self.trees_data = []
        self.sampling_with_replacement(x, y)
        for i in range(self.n_estimators):
            Tree=Decision_Tree(max_depth=self.max_depth)
            Tree.fit(self.trees_data[i][0],self.trees_data[i][1])
            self.trees.append(Tree)
        return self

    def predict(self, x):
        answer=[]
        for j in range(x.shape[0]):
            prediction=[]
            for i in range(self.n_estimators):
                prediction.append(self.trees[i].predict(x[j:j+1])[0])
            answer.append(max(prediction,key=prediction.count))
        return np.array(answer)



if __name__ == '__main__':
    x=np.random.random(size=(100,3))
    y=np.random.randint(size=(100,),high=100,low=10)

    x_test=np.random.random(size=(100,3))
    model = RandomForest(n_estimators=100, max_depth=8, same_rate=0.5)
    model.fit(x,y)
    A=model.predict(x_test)
    print(A)
