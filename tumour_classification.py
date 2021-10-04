import lightgbm as lgb  
import pandas as pd  
import numpy as np  
from sklearn.metrics import roc_auc_score  
import sklearn    
import matplotlib.pyplot as plt     
from sklearn import model_selection
from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_curve
from sklearn.metrics import auc 
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

L=['BLCA','BRCA','CESC','COAD','HNSC'
  ,'KIRC','LIHC','LUAD','LUSC','OV'
  ,'READ','SKCM','STAD','THCA','UCEC']
  



def readdata(name):
    name=name
    dataset = pd.read_csv(name,sep=',',header=None)
    return dataset


if __name__ == "__main__":

    dataset=readdata("15patient_feature.txt")
    
    for i in range(1,1281):
        dataset[i] = round(dataset[i],5)

    zz=[]
    dataset=dataset.sort_values(by=1281 , ascending=True,axis=0)
    dataset=dataset.reset_index()
    y = label_binarize(dataset[1281], classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    u=pd.DataFrame(y)
    u.columns=[x for x in range(1282,1297)]
    y1=dataset.join(u)
    del y1['index']
    

    dataset = shuffle(dataset)
    y1 = shuffle(y1)
    plt.figure(figsize=(12,15))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    for i,j in zip(range(1,16),L):
        array= y1.values   
        X= array[:,1:1281]       
        Y= array[:,1281+i]         
        validation_size= 0.2   
        seed= 10      
        X_train, Y_train, X_validation, Y_validation= model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
       
        clf1 = lgb.LGBMClassifier(
                boosting_type='dart', num_leaves=41, reg_lambda=1,xgboost_dart_mode=True,
                max_depth=-1, n_estimators=1200, objective='binary',metric= 'auc',bagging_fraction=0.8,
                subsample=0.8, colsample_bytree=0.8, subsample_freq=1,lambda_l1= 0.6,lambda_l2= 0,
                learning_rate=0.01, min_child_weight=1,feature_fraction=0.8
            )
        clf1.fit(X_train, X_validation.astype('int'), eval_set=[(X_train, X_validation),(Y_train, Y_validation)], eval_metric='logloss',early_stopping_rounds=300)
        pre11=clf1.predict_proba(Y_train)
        pre111=clf1.predict(Y_train)

        fpr,tpr,threshold = roc_curve(Y_validation.astype('int'),pre11[:,1].ravel())
        #auc1=roc_auc_score(Y_validation.astype('int'),pre111)
        auc1=auc(fpr, tpr)
        #auc1=metrics.auc(fpr, tpr)
        L1.append(auc1)



        plt.subplot(5,3,i)
        plt.plot(fpr, tpr, color='darkorange',label='AUC = {0:0.2f}'
                       ''.format(auc1),linewidth=2.5 )
        plt.plot([0, 1], [0, 1], 'k--',color='black',linewidth=1)
        plt.legend(loc="lower right",frameon=False)
        plt.text(0.8, 0.4, j,size=15)
        #plt.xticks([])
        #plt.yticks([])
        plt.tick_params(labelsize=8)
        #plt.title(j,fontsize=14,y=1.03)
    plt.tight_layout()
    plt.savefig("gene_neighbor_1280.pdf",bbox_inches='tight',dpi=400,pad_inches=0.0)

    plt.show()
