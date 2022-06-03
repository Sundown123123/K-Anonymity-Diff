import pandas as pd
import numpy as np

names = (
    'age',
    'workclass', 
    'fnlwgt', 
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income',
)

df = pd.read_csv("adult.data.txt", sep=", ", header=None, names=names, index_col=False, engine='python')
avg_age_1=df['age'].mean()
avg_age_1#原数据集平均年龄
#敏感度
print("敏感度:",avg_age_1-(avg_age_1*32561-90)/32560)
dfn = pd.read_csv("out.csv", sep=",", index_col=False, engine='python')
dfn=dfn.iloc[:,[1,-1]]

#加噪声
def add_laplace_noise(data_list, μ=0, b=1):
    laplace_noise = np.random.laplace(μ, b, len(data_list)) # 为原始数据添加μ为0，b为1的噪声
    res=np.append((laplace_noise+data_list.T[0]),data_list.T[1]).T
    return res.reshape(2,len(data_list)).T
def age_mean(a):
    sage=0
    scnt=0
    for row in a:
        sage+=row[0]*row[1]
        scnt+=row[1]
    return sage/scnt

data1 = dfn.to_numpy()
data=data1
print("原始无噪声数据|均值：" + str(age_mean(data)))
x1=age_mean(data)
noise_list = add_laplace_noise(data)
print("加噪声后的数据|均值：" + str(age_mean(noise_list)))
y1=age_mean(noise_list)
data=np.delete(data,-1,axis=0)
print("原始无噪声数据|均值：" + str(age_mean(data)))
x2=age_mean(data)
noise_list = add_laplace_noise(data)
print("加噪声后的数据|均值：" + str(age_mean(noise_list)))
y2=age_mean(noise_list)

#差分隐私分析
ans1=(x1*32561-x2*32556)/5
ans2=(y1*32561-y2*32556)/5
print("根据原数据集/K匿名数据集的推出删除的数据:",ans1)
print("根据加入噪声后K匿名数据集的推出删除的数据:",ans2)