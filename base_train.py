import sklearn.datasets
from sklearn.datasets import load_iris
from sklearn import model_selection
import graphviz
from sklearn import tree
import numpy as np
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from random import random
from random import randint

class RU:
    def __str__(self):
        return "<RU.Class>"
    def __len__(self):
        return len(self.Classifier)
    def __init__(self,loadData=None):
        if loadData!=None:
            print("数据已读取")
            self = loadData
        self.allLabel = []
        self.Classifier = []
        self.weightsData = []
        pass
    def feature_names(self,feature_names):
        self.feature_names = feature_names

    def weights(self,model=0,weightsData=None):
        if model==0:
            return self.weightsData
        if model==1:
            weightsData = [1 for i in range(0, len(self))]
            self.weightsData = np.array(weightsData)
            return self.weightsData
        if model==2:
            if len(weightsData)==len(self):
                self.weightsData = np.array(weightsData)
            else:
                raise ValueError(f'输入的权重与已有的结构不匹配,结构权重需求:{len(self)},输入的权重为：{len(weightsData)}')
            return self.weightsData
        if model==3:
            with open(weightsData, 'rb') as f:
                loaded_data = pickle.load(f)
            self.weightsData = loaded_data["weightsData"]
            return self.weightsData
        return 0




    def fit(self,data):
        self.allLabel = [None for i in range(0, len(data[0]))]
        data = np.array(data)
        self.data = data
        Y = deepcopy(data[:,len(data[0])-1])
        print(Y)
        le = LabelEncoder()
        for i in range(0, len(data[0])):
            self.allLabel[i] = deepcopy(le.fit(data[:, i]))
        print(self.allLabel)

        for i in range(0, len(data[0])):
            data[:, i] = self.allLabel[i].transform(data[:, i])

        print(data)

        resultZIP = list(set(zip(data[:, len(data[0])-1],Y)))
        resultDict = dict()
        for item in resultZIP:
            resultDict[int(item[0])] = item[1]
        self.resultDict = resultDict

        X_trainer, X_test, Y_trainer, Y_test = model_selection.train_test_split(data[:,0:len(data[0])-1], data[:,len(data[0])-1], test_size=0.3)
        self.X_trainer = np.array(X_trainer, dtype=float)
        self.X_test = np.array(X_test, dtype=float)
        self.Y_trainer = np.array(Y_trainer, dtype=float)
        self.Y_test = np.array(Y_test, dtype=float)


    def KNeighbors(self,k:int):
        tmp = {"name":"KNeighbors","weight": 1,"classifier":None}
        clf =KNeighborsClassifier(n_neighbors=k)
        clf.fit(self.X_trainer, self.Y_trainer)
        tmp["classifier"] = clf
        self.Classifier.append(deepcopy(tmp))


    def DecisionTree(self):
        tmp = {"name": "DecisionTree", "weight": 1, "classifier": None}
        clf = tree.DecisionTreeClassifier(criterion='entropy')
        clf.fit(self.X_trainer, self.Y_trainer)
        tmp["classifier"] = clf
        self.Classifier.append(deepcopy(tmp))

    def BernoulliNB(self):
        tmp = {"name": "BernoulliNB", "weight": 1, "classifier": None}
        clf = BernoulliNB()
        clf.fit(self.X_trainer, self.Y_trainer)
        tmp["classifier"] = clf
        self.Classifier.append(deepcopy(tmp))

    def run(self):
        for item in self.Classifier:
            print(item)
            clf_Score = item["classifier"].fit(self.X_trainer,self.Y_trainer)
            print("模型在测试集上进行评分：\n",clf_Score.score(self.X_test,self.Y_test))

    def resultToClass(self,res):
        result = []
        for i in res:
            result.append(self.resultDict[i])

        return np.array(result)

    def resultToStrClass(self,res):
        return self.resultDict[res]

    def fit_transform(self,inputData):
        #input = [['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑']]
        datanew = np.array(inputData)
        for i in range(0, len(datanew[0])):
            datanew[:, i] = self.allLabel[i].transform(datanew[:, i])
        return np.array(datanew, dtype=float)

    def merge(self):
        pass

    def generate_random_list(self,n):
        # 生成n个随机数
        random_list = np.random.rand(n)

        # 将随机数归一化，使其和为1
        normalized_list = random_list / np.sum(random_list)

        return normalized_list

    def basePredict(self,inputData:list):
        inputData = self.fit_transform(inputData)
        res = []
        for item in self.Classifier:
            tmp = item["classifier"].predict(inputData)
            res.append(int(tmp[0]))
        return np.array(res)

    def Predict(self,inputData):
        print(inputData)
        res = []
        for item in self.Classifier:
            tmp = item["classifier"].predict([inputData])
            res.append(int(tmp[0]))
        return np.array(res)

    def RandomTree(self,n:int):
        for _ in range(0,n):
            num = randint(0, 3)
            if num == 0:
                self.KNeighbors(randint(0,4))
            if num == 1:
                self.DecisionTree()
            if num == 2:
                self.BernoulliNB()
        print(f"已成功构建{n}个随机树")

    def saveModel(self,path):
        print("EE")
        model = {"allLabel":self.allLabel,"classifier":self.Classifier,"weights":self.weightsData}
        data = {"data":self}
        with open(f"{path}", 'wb') as f:
            pickle.dump(data, f)
            print("保存完毕")
        return 0

    def weightsPredict(self,inputData:list):
        result = self.basePredict(inputData)
        print(result)
        res = self.resultToClass(result)
        weights = self.weights(model=0)
        weightsRes = dict()
        res = dict()
        for i in range(0,len(self)):
            weightsRes[result[i]]=0
        for i in range(0,len(self)):
            weightsRes[result[i]] += weights[i]

        for item in weightsRes:
            #print(self.resultToClass(np.array([item])))
            #print(weightsRes[item])
            res[self.resultToStrClass(item)] = weightsRes[item]

        return res
        # print(result)
        # print(weights)
        # print(weightsRes)

    def find_max_key(self,dictionary):
        max_key = max(dictionary.items(), key=lambda x: x[1])[0]
        return max_key

    def trainWeights(self):
        allResult = []
        while True:
            randomWeights = self.generate_random_list(len(self))
            for data in self.X_trainer:

                result = self.Predict(data)
                print(result)
                res = self.resultToClass(result)
                weights = self.weights(model=0)
                weightsRes = dict()
                res = dict()
                for i in range(0,len(self)):
                    weightsRes[result[i]]=0
                for i in range(0,len(self)):
                    weightsRes[result[i]] += randomWeights[i]

                for item in weightsRes:
                    #print(self.resultToClass(np.array([item])))
                    #print(weightsRes[item])
                    res[self.resultToStrClass(item)] = weightsRes[item]

                allResult.append(res)
            print(allResult)
            tmpMax = []
            for item in allResult:
                tmpMax.append(self.find_max_key(item))
            tmpMax = np.array([tmpMax])

            for i in range(0,len(tmpMax)):
                tmpMax[i] = self.allLabel[len(self.data[0])-1].transform(tmpMax[i])
            tmpMax = np.array(tmpMax,dtype=float)
            tmpMax = tmpMax[0]
            print(tmpMax)
            print(self.Y_trainer)
            break
        return res

    def train(self):
        X = self.X_trainer
        Y = self.Y_trainer
        result = []
        standard = []
        for pieData in X:
            tmpData = []
            for item in self.Classifier:
                tmp = item["classifier"].predict([pieData])
                tmpData.append(tmp[0])
            result.append(tmpData)
        print(result)
        for i in Y:
            standard.append([i])
        print(standard)
        print(X)
        print(Y)
        print(result)
        # while True:
        #     for i in
        #     weightsData = self.generate_random_list(len(self))
        #     resultData = weightsData.dot(result[0])
        # print("EEND")
        # print(weightsData)
        # print("EEEEE")
        # print(result)
        #
        # print(resultData)
        # result = self.basePredict(X)
        # print(result)
        # res = self.resultToClass(result)
        # weights = self.weights(model=0)
        # weightsRes = dict()
        # res = dict()
        # for i in range(0,len(self)):
        #     weightsRes[result[i]]=0
        # for i in range(0,len(self)):
        #     weightsRes[result[i]] += weights[i]
        #
        # for item in weightsRes:
        #     #print(self.resultToClass(np.array([item])))
        #     #print(weightsRes[item])
        #     res[self.resultToStrClass(item)] = weightsRes[item]
        # print(res)
        # quit()
        # return res
        # # print(result)
        # # print(weights)
        # # print(weightsRes)





# data = sklearn.datasets.load_iris()
# print(data)
# quit()




r = RU()

data =  [
            ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
            ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
            ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
            ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
            ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
            ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '是'],
            ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '是'],
            ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '是'],
            ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '否'],
            ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '否'],
            ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '否'],
            ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '否'],
            ['浅白', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '否'],
            ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '否'],
            ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '软粘', '否'],
            ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否'],
            ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '否'],
            ['浅白', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '否'],
            ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '否'],
            ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '软粘', '否'],
            ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否'],
            ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '否']
        ]

print(len(r))
r.fit(data)
r.RandomTree(5)
r.saveModel('model.rth')

r.weights(model=1,weightsData=[0.2,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3])

result = r.weightsPredict([['青绿', '硬挺', '清脆', '模糊', '平坦', '硬滑']])
print(result)





# res = r.basePredict([['青绿', '蜷缩', '浊响', '清晰', '平坦', '硬滑']])
# print(r.resultToClass(res))
#r.run()
