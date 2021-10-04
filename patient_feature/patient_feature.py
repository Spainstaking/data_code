import numpy as np
import pandas as pd
import os  

ttt = pd.read_csv("struc2vec/emb/1Human_Interactome_128.txt",sep=' ',header=None)

a1=[]
a=[]
def file_name(file_dir):   
    for root, dirs, files in os.walk(file_dir):   
        for i in files:
            a1.append(i)  
    
file_name('dataset\Mutation_Individual')
for i in a1:
    a.append('dataset/Mutation_Individual/'+i)
    
b=[]
for i in a1:
    b.append('processing/in_data/'+i)

for i,j,k in zip(a,range(1,16),b):
    data = pd.read_csv(i,sep='\t',header=None)
    data2=data.merge(ttt,left_on=data.iloc[:,1],right_on=ttt.iloc[:,0],how='left')
    print(data2.head())
    data2=data2.drop('key_0',axis=1)
    data2['0_y']=j
    
    data2.to_csv(k,index=0,header=None)
    
ttt = pd.read_csv("patient_feature/gene_neighbor.txt",sep=',',header=None)

for i in b:
    z = pd.read_csv(i,sep=',',header=None)
    data2=z.merge(ttt,left_on=z.iloc[:,1],right_on=ttt.iloc[:,0],how='left')
    c = data2.pop('1_y')
    data2.insert(4,'c_new',c)
    #data2[3]=data2['1_y']
    data2=data2.drop(['key_0','0_y'],axis=1)
    print(data2)
    data2.to_csv(i,index=0,header=None)
    
b=[]
for i in a1:
    b.append('processing/in_data/'+i)
    
bb=[]
for i in a1:
    bb.append('processing/out_data/1_10_'+i)
    
for m,n in zip(b,bb):
    data = pd.read_csv(m,sep=',',header=None)
    #data=data.drop(132,axis=1)
    data=data.dropna(axis=0, how='any')
    data=data.sort_values(by=3)
    data['c']=1
    for i in range(4,132):
        data[i] = data[i].astype('float64')
    q=[2.0,5.0,9.0,14.0,21.0,29.0,42.0,64.0,126.0]
    w=[2,3,4,5,6,7,8,9,10]
    for i,j in zip(q,w):
        data.loc[data[3]>i,'c']=j
    print(data.head())
    data=data.fillna(0)
    df = data.groupby([data[0],data['c']]).sum()
    print(df.head())
    df1=df.drop([1,2,3],axis=1)
    df1.to_csv('group.txt',header=None)
    df11=pd.read_csv("group.txt",sep=',',header=None)
    
    LL=[]
    for i in range(0,len(list(set(df11[0])))):
        for j in range(0,10):
            LL.append(list(set(df11[0]))[i])
    LL1=[]
    for i in range(0,len(list(set(df11[0])))):
        for j in range(1,11):
            LL1.append(j)
    dd=np.array(LL).T
    dd1=pd.DataFrame(dd)
    dd1[1]=LL1
    df0=pd.merge(dd1,df11,on = [0,1],how='left')
    df0=df0.fillna(0)
    p1=df0.iloc[0::10,:]
    p2=df0.iloc[1::10,:]
    p3=df0.iloc[2::10,:]
    p4=df0.iloc[3::10,:]
    p5=df0.iloc[4::10,:]
    p6=df0.iloc[5::10,:]
    p7=df0.iloc[6::10,:]
    p8=df0.iloc[7::10,:]
    p9=df0.iloc[8::10,:]
    p10=df0.iloc[9::10,:]
    q=np.hstack((p1,p2,p3,p4,p5,p6,p7,p8,p9,p10))
    zz=pd.DataFrame(q)
    print(zz)
    for i in range(129,1170,130):
        zz=zz.drop(i,axis=1)
    for i in range(130,1171,130):
        zz=zz.drop(i,axis=1)
    zz=zz.drop(1,axis=1)
    #[1,32,33,64,65,96,97,128,129]
    zz.to_csv(n,header=None,index=0)
    
    
a1=[] 

file_name('processing/out_data')

zz=[]
for i in a1:
    zz.append('processing/out_data/'+i)
    
df=pd.DataFrame()

for i,j in zip(zz,range(1,16)):
    data = pd.read_csv(i,sep=',',header=None)
    z=pd.DataFrame(data)
    z['c']=j
    df=pd.concat([df,z], axis=0, ignore_index=True)
    
df.to_csv('15patient_feature.txt',index=0,header=None)