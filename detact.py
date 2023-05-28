from sklearn.datasets import load_iris
from sklearn import model_selection
import graphviz
from sklearn import tree
import numpy as np
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy
import pickle
from base_train import RU

with open('model.rth', 'rb') as f:
    loaded_data = pickle.load(f)

r = RU(loadData=loaded_data["data"])
result = r.weightsPredict([['青绿', '硬挺', '清脆', '模糊', '平坦', '硬滑']])
print(result)

quit()
