import numpy as np


"""--------决策树算法实现
    1.基于cart决策树 利用基尼不纯值对数据进行分裂
            先完成分类任务 

"""
class TreeNode():
    def __init__(self,feature=None,threshold=None,left=None,right=None,value=None):
        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value

class Decision_Tree():
    def __init__(self,max_depth=8,min_samples_split=2,min_samples_leaf=1,ccp_alpha=None):
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.min_samples_leaf=min_samples_leaf
        self.ccp_alpha=ccp_alpha
        self.root=None

    def get_unique_label(self,x):   #获取单列特征或样本的 类别,及对应的个数
        values, count = np.unique(x, return_counts=True)
        dic = dict(zip(values, count))
        return values,dic           #返回类别列表 ,以及对应类别的字典
    """-------计算目标基尼值----------
        公式:基尼值 = 1 - (类别1占比的平方 + 类别2占比的平方 + ... + 类别k占比的平方)
    """
    def calculate_gini(self,y,num_sample):
        label_list,dic=self.get_unique_label(y)               #获取y的字典 统计类别个数计算类别占比
        summary=0
        for i in label_list:
            summary+=dic[i]*dic[i]
        return 1-summary/(num_sample*num_sample)

    def calculate_Gini_Coefficient(self,y,n_dimension,left_index,right_index):
        """
            feature_label_col 传入的是dataframe     DF=df[feature_col,label_col]   废案 如果要创建多叉数的话可以这么使用
            直接传入 y即可
            n_dimension:为num_sample 为该父节点的样本个数
            left_index 左节点的索引数组
            right_index 右节点的索引数组
            由于多叉树易造成过拟合  现 重构成二叉树     传入左右字数索引列表即可

            对特征内的类别特征进行计算
            计算基尼指数 作为分裂条件
            公式: (左样本/总样本)*左节点基尼值+(右样本/总样本)*右节点基尼值
            实际上就是在特征列中每种状态对分类类别的影响 所以要乘于权重
         :return:
        """
        left_len=len(left_index)
        right_len=len(right_index)
        gini1=self.calculate_gini(y[left_index],left_len)
        gini2=self.calculate_gini(y[right_index],right_len)
        summary=(gini1*left_len+gini2*right_len)/n_dimension
        return summary
    def best_split(self, x, y):
        """
            决策函数 将特征列中各个分类进行各个计算得出最佳决策点 最优基尼指数


        :param x:
        :param y:
        :return:
        """
        if x<self.min_samples_leaf:
            return None,None,None
        best_gini=np.inf
        best_feature=None
        best_threshold=None

        num_feature=x.shape[1]
        num_sample=x.shape[0]

        for feature in range(num_feature):
            thresholds = []
            # thresholds=np.unique(x[:,feature]) 这样写容易冗余做一些多余判断
            list_=np.unique(x[:,feature])
            size=len(list_)
            for i in range(size-1):
                thresholds.append((list_[i]+list_[i+1])/2)

            for threshold in thresholds:
                left=np.where(x[:,feature]<=threshold)[0]
                right=np.where(x[:,feature]>threshold)[0]
                if len(left)==0 or len(right)==0:
                    continue
                gini_coefficient=self.calculate_Gini_Coefficient(y,num_sample,left,right)
                if gini_coefficient < best_gini:
                    best_gini = gini_coefficient
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gini

    def build_tree(self,x,y,depth):
        """
            基于上述代码实现 构建一个 二叉树达成决策树的主要功能框架
        :param x:
        :param y:
        :return:
        """

        if len(np.unique(y))==1:
            return TreeNode(value=y[0])

        values, counts = np.unique(y, return_counts=True)
        majority_class = values[np.argmax(counts)]

        if x.shape[0]<self.min_samples_split:
            return TreeNode(value=majority_class)

        if depth>=self.max_depth:
            return TreeNode(value=majority_class)

        best_feature, best_threshold, best_gini = self.best_split(x, y)

        if best_feature is None:
            return TreeNode(value=majority_class)
        #获取左右子树的特征索引
        left_index=np.where(x[:,best_feature]<=best_threshold)[0]
        right_index=np.where(x[:,best_feature]>best_threshold)[0]
        #后序递归构建二叉树
        left_sub=self.build_tree(x[left_index],y[left_index],depth+1)
        right_sub=self.build_tree(x[right_index],y[right_index],depth+1)

        return  TreeNode(feature=best_feature,threshold=best_threshold
                         ,left=left_sub,right=right_sub)


    """
        fit训练函数直接创建一个完整的二叉树即可
        后续直接对样本进行预测即可
    """
    def fit(self,Feature,label):
        Feature=np.array(Feature)
        label=np.array(label)
        self.root=self.build_tree(Feature,label,0)
        return self

    def predict(self,Feature):
        if self.root is None:
            return
        predictions=np.empty(len(Feature))
        Feature=np.array(Feature)
        for i,x in enumerate(Feature):
            predictions[i]=self._predict_single(x,self.root)
        return predictions

    def _predict_single(self, x, node):
        """
        预测单个样本  用递归进行预测
        x: shape=(n_features,) 一维数组
        node: 当前节点
        """
        if node.value is not None:
            return node.value

        # 根据特征值判断走左子树还是右子树
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)


if __name__=='__main__':
     pass












