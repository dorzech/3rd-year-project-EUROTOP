# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 19:49:44 2022

@author: Dorota Orzechowska
"""
# Import libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


from sklearn.ensemble import RandomForestRegressor

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

random_state = 134

# Load the data 
df= pd.read_excel('Database_Eurotop_v5.xlsx')

#dfx= pd.read_excel('Database_Eurotop_v5.xlsx')
df.head()

#create delete list for parameters unused
delete_list = ['Label']

#delete non-core data
df.drop(df.index[df['Core data'] != 'Z'], inplace=True)

#delete rows with Nan values of'q'
df = df.dropna(subset=['q'])

# delete simulation with RF or CF = 4 or 3
df.drop(df.index[df['RF'] == 4], inplace=True)
df.drop(df.index[df['CF'] == 4], inplace=True)
df.drop(df.index[df['RF'] == 3], inplace=True)
df.drop(df.index[df['CF'] == 3], inplace=True)

dfsave = pd.DataFrame({
    'Label':df['Label'],
    'm':df['mmm'],
    'h':df['h'],
    'Hm0 toe':df['Hm0 toe'],
    'Tm1,0t':df['Tm1,0t'],
    'β':df['β'],
    'ht':df['ht'],
    'Bt':df['Bt'],
    'hb':df['hb'],
    'B':df['B'],
    'cotad':df['cotαd'],
    'cotaincl':df['cotαincl'],
    'cotau':df['cotαu'],
    'gf_d':df['gf_d'],
    'gf_u':df['gf_u'],
    'gf':df['gf'],
    'D':df['D'],
    'D50_d':df['D50_d'],
    'D50_u':df['D50_u'],
    'Ac':df['Ac'],
    'Rc':df['Rc'],    
    'Gc':df['Gc'],
    'Kr':df['Kr'],
    'Kt':df['Kt'],
    'q' :df['q']
    })



#delete 'Label' to change data type for float
for x in delete_list:
    dfsave = dfsave.drop(columns=[x])

# convert in float
#dfsave.drop(dfsave.head(1).index, inplace=True)
dfsave = dfsave.astype('float')
dfsave = dfsave.dropna(subset=['m'])
dfsave = dfsave.dropna(subset=['D'])
a = dfsave.describe()



#delete q == 0
dfsave.drop(dfsave.index[dfsave['q'] == 0 ], inplace=True)


#%matplotlib qt

for i in dfsave.columns:
    fig = plt.hist(dfsave[i],bins=10)
    plt.title(i)
    plt.show(fig)

# delete the NaN 
#dfsave = dfsave.dropna()
#a = df.describe()

#corr matrix to see correlation of parameters
corr_matrix = dfsave.corr()
f, ax = plt.subplots(figsize=(20,20))
g = sns.heatmap(corr_matrix, cmap = sns.diverging_palette(10, 250, as_cmap = True), annot = True)

#creating an additional feature - compute spectral wave length in deep water lm0 in meter
dfsave['Lm0'] = ((9.81*(dfsave['Tm1,0t'])**2)/(2*np.pi))

# create dataframe containing the model parameters
dff = pd.DataFrame({
    'Lm0' :dfsave['Lm0'],
    'Hm0' :dfsave['Hm0 toe'],
    'Hm0/Lm0':dfsave['Hm0 toe']/dfsave['Lm0'],
    'β':dfsave['β'],
    'h/Lm0':dfsave['h']/dfsave['Lm0'],
    'ht/Hm0':dfsave['ht']/dfsave['Hm0 toe'],
    'Bt/Lm0':dfsave['Bt']/dfsave['Lm0'],
    'hb/Hm0':dfsave['hb']/dfsave['Hm0 toe'],
    'B/Lm0':dfsave['B']/dfsave['Lm0'],
    'Ac/Hm0':dfsave['Ac']/dfsave['Hm0 toe'],
    'Rc/Hm0':dfsave['Rc']/dfsave['Hm0 toe'],
    #'Rc':dfsave['Rc'],
    'Gc/Lm0':dfsave['Gc']/dfsave['Lm0'],
    'm':dfsave['m'],
    'cotad':dfsave['cotad'],
    'cotaincl':dfsave['cotaincl'],
    'gf':dfsave['gf'],
    'q':dfsave['q'],
    'D/Hm0':dfsave['D']/dfsave['Hm0 toe'],
    #target:
    'qAD':np.log10(dfsave['q']/np.sqrt(9.81*(dfsave['Hm0 toe'])**3)),
    })
#dff.head()

dff.describe()

#dff = dff.dropna()

b= dff.describe()

# Replace infinite updated data with nan
dff.replace([np.inf, -np.inf], np.nan, inplace=True)
# Drop rows with NaN
dff.dropna(inplace=True)

#dfsave.to_csv('NN_OVERTOPPING6.csv', decimal=".", sep=' ', header = False, index = False ) 
#val_=dfsave.describe()

corr_matrix = dff.corr()
f, ax = plt.subplots(figsize=(20,20))
g = sns.heatmap(corr_matrix, cmap = sns.diverging_palette(10, 250, as_cmap = True), annot = True)


train_set, test_set = train_test_split(dff, test_size=0.2, random_state = random_state)

#scaler_std = StandardScaler()
scaler = StandardScaler()

ANN_1 = dfsave.loc[test_set.index] 
ANN_1_q = pd.DataFrame(ANN_1['q'])
ANN_1_qad = pd.DataFrame(np.log10(ANN_1['q']/np.sqrt(9.81*(ANN_1['Hm0 toe'])**3)))

ANN_1 = ANN_1.drop(['Lm0', 'D', 'gf','Kr', 'Kt','q', 'cotaincl'], axis = 1)



#prepare file for ANN tool
ANN_1['Kr'] = 0
ANN_1['Kt'] = 0
ANN_1['q'] = 1
ANN_1.insert(loc=0, column='Label', value=df['Label'])
ANN_1.to_csv('ANN_1.csv', decimal=".", sep='|', index = False ) 

nn_overtoppingANN_1 = pd.read_csv('ANN_results.csv', decimal=".", sep='|')

nn_overtoppingANN_1_transform = pd.DataFrame(nn_overtoppingANN_1['Average'])

ANN_1_Hm = pd.DataFrame(ANN_1["Hm0 toe"])
ANN_1_Hm=ANN_1_Hm.reset_index()
ANN_1_Hm = ANN_1_Hm.drop(['index'], axis=1)

nn_overtoppingANN_1_transform_qad = pd.DataFrame(np.log10(nn_overtoppingANN_1_transform['Average']/np.sqrt(9.81*(ANN_1_Hm['Hm0 toe'])**3)))


# save description of train data
train_transform = pd.DataFrame(train_set.drop(['Lm0', 'Hm0','q','qAD'], axis=1))
desc_train_transform = train_transform.describe()
desc_train_transform.to_excel('desc_train_transform.xlsx') 
train_transform = pd.DataFrame(scaler.fit_transform(train_transform))
train_transform.columns = train_set.drop(['Lm0','Hm0','q','qAD'], axis=1).columns

test_transform = pd.DataFrame(test_set.drop(['Lm0','Hm0','q','qAD'], axis=1))

#save description of test data
desc_test_transform = test_transform.describe()
desc_test_transform.to_csv('desc_test_transform.csv', decimal=".", sep=' ', header = True, index = False ) 

#add noise i zapisz ponownie
#jesli dla zaszumionego datasetu będzie podobny wynik albo mniejszy - to jest ok, a jesli wiekszy to overfitting i 
test_transform = pd.DataFrame(scaler.transform(test_transform))
test_transform.columns = test_set.drop(['Lm0','Hm0','q','qAD'], axis=1).columns

fin_ANN_1_qad_test_transform = pd.DataFrame(scaler.fit_transform(ANN_1_qad.values.reshape(-1,1)))

fin_nn_overtoppingANN_1_transform_qad = pd.DataFrame(scaler.transform(nn_overtoppingANN_1_transform_qad.values.reshape(-1,1)))




for i in train_transform.columns:
    fig = plt.hist(train_transform[i],bins=20)
    plt.title('Transformed '+i)
    plt.show(fig)

#organize the data 
train = {
    'features': train_transform,
    'target' : scaler.fit_transform(train_set['qAD'].values.reshape(-1,1))
}
#targetu się nie transformuje
test = {
    'features': test_transform,
    'target' : pd.DataFrame(scaler.fit_transform(test_set['qAD'].values.reshape(-1,1)))
}

test['features'].to_csv('ANN_last.csv', decimal=".", sep='|', index = False ) 

train['features'].to_csv('train_features.csv', decimal=".", sep=' ', header = True, index = False ) 

# keep Hm0 data for the inverse_transform
Hm0_test = pd.DataFrame(test_set['Hm0'])


# select the metrics
scoring = ['neg_mean_squared_error','r2']


def transform(Hm0, data,scaler):
    prediction = scaler.inverse_transform(data.values.reshape(-1,1))
    q_overtopping = (10**(prediction.flatten()))*np.sqrt(9.81*(Hm0_test['Hm0'])**3)
    return q_overtopping






#from sklearn.model_selection import GridSearchCV
# RF Regressor

### Add validation data set
# split the train set between train and validation set
train_feature_rf, train_val_feature_rf, train_target_rf, train_val_target_rf = train_test_split(train['features'], train['target'], test_size=0.2,random_state=random_state)
train['features_rf'] = train_feature_rf
train['target_rf'] = train_target_rf
train['val_feature_rf'] = train_val_feature_rf
train['val_target_rf'] = train_val_target_rf


#desc_train_feature_rf = pd.DataFrame(train_feature_rf).describe()
#desc_train_target_rf = pd.DataFrame(train_target_rf).describe()
#desc_train_val_feature_rf = train_val_feature_rf.describe()
#desc_train_val_target_rf = pd.DataFrame(train_val_target_rf).describe()

# build function for hyperparameters search

def RandomForestR(params,  scoring, features, target):

    rf = RandomForestRegressor(random_state)
   
    for x in scoring:
        grid_search = GridSearchCV(estimator = rf, param_grid = params, 
                                   n_jobs = -1, verbose = 2, scoring=x)
       
        grid_search.fit(features, target)
        print()
        print("######### Tuning hyper-parameters for %s" % x)
        print()
        print()
        print("Best parameters set found on development set:")
        if x == 'neg_mean_squared_error':
            print(grid_search.best_params_, ' -----> SCORE RMSE: %f' %
                  np.sqrt(-grid_search.best_score_))
        else:
            print(grid_search.best_params_, ' -----> SCORE: %f' %
                  grid_search.best_score_)
# search hyperparameters

scoring = ['neg_mean_squared_error', 'r2']
           

# Create the parameter grid based on the results of random search 
params_grid = {
    #'bootstrap': [True],
    'max_depth': [ 25],
    #'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [15, 20, 25],
    #'min_samples_split': [6, 32],
    'n_estimators': [176, 199]
}



RandomForestR(params_grid, scoring, train['features'], train['target'].ravel())


### Search the best n_estimator
# search best n_estimators

sum_of_errors=[]
for i in range(1, 201):
    print(i)
    rf = RandomForestRegressor( max_depth=25, n_estimators=i)
    rf.fit(train['features_rf'],train['target_rf'])
    #errors = [mean_squared_error(train['val_target_rf'], prediction)
    #     for prediction in rf.predict(train['val_feature_rf'])]
    errors = []
    for real, pred in zip(train['val_target_rf'], rf.predict(train['val_feature_rf'])):
        try:
            errors.append(mean_squared_error(real, [pred], squared = False))    
        except:
            print(real, [pred])
            #
    sum_of_errors.append(np.mean(errors))

bst_n_estimators = np.argmin(sum_of_errors) + 1
bst_n_estimators =195
#ABS module

rf_best = RandomForestRegressor(max_depth=25, n_estimators=bst_n_estimators, random_state = random_state)
rf_best.fit(train['features_rf'], train['target_rf'])

#plot validation error for the n_estimators
min_error = np.min(errors)
plt.figure(figsize=(20, 10))
plt.plot(np.arange(1, len(errors) + 1), errors, "b.-")
plt.plot([bst_n_estimators, bst_n_estimators], [0, min_error], "k--")
plt.plot([0, 120], [min_error, min_error], "k--")
plt.plot(bst_n_estimators, min_error, "ko")
plt.axis([0, 500, 0, 1])
plt.xlabel("N_estimators")
plt.ylabel("Mean error", fontsize=16)
plt.title("Mean error vs n_estimators for RF model", fontsize=14)

#plot sum of errors
plt.figure(figsize=(20, 10))
plt.plot(sum_of_errors)
plt.ylabel("Validation error", fontsize=16)
plt.xlabel("N_estimators")
plt.title("Learning curve showing validation error for RF model", fontsize=14)

###  Feature importances from t model
# check the feature importance
 
def features_importances(estimator, features):
    importances = estimator.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(15, 10))
    plt.title("Feature importances")
    plt.bar(range(len(importances)),
            importances[indices], color="blue", align="center")
    plt.xticks(range(len(importances)),
               features.columns[indices], fontsize=10, rotation=90)
    plt.xlim([-1, len(importances)])
    plt.show()
    
features_importances(rf,train['features'])

#print(rf.feature_importances_)

### Build the rf model and make predictions
# run rf with best parameters
rf = RandomForestRegressor(     max_depth=25,
                                n_estimators=195,
                                random_state=random_state)
rf.fit(train['features'], train['target'].ravel())

prediction = rf.predict(test['features'])
test['predictions'] = pd.DataFrame(prediction)
print("############## SCORE DATASET TEST FOR rf ########")
print()
print("R2 TEST")
print(r2_score(test['target'], test['predictions']))
print()
print("RMSE TEST")
print(np.sqrt(mean_squared_error(test['target'], test['predictions']))) 



print("############## SCORE DATASET TEST FOR ANN ########")
print()
print("R2 TEST")
print(r2_score(fin_ANN_1_qad_test_transform, fin_nn_overtoppingANN_1_transform_qad))
print()
print("RMSE TEST")
print(np.sqrt(mean_squared_error(fin_ANN_1_qad_test_transform, fin_nn_overtoppingANN_1_transform_qad)))

plt.rcParams["figure.figsize"] = [10.50, 7.50]
plt.rcParams["figure.autolayout"] = True
plt.title('Observations vs Predictions for RF model')
plt.xlabel("Q measured m3/s/m"), plt.ylabel("Q predictions RF m3/s/m")
plt.scatter(transform(Hm0_test, test['target'],scaler),
            transform(Hm0_test, test['predictions'],scaler),
            s = 10,marker = '.',c='red',
           )
plt.plot(np.linspace(0, 1),
         np.linspace(0, 1), ':k')
#plt.axis([0.00000001, 1, 0.00000001, 1])
plt.loglog()
plt.grid()
# rf model vs ANN model



#Compare the score and predictions from theses models
rf = RandomForestRegressor(     max_depth=25,
                                n_estimators=bst_n_estimators,
                                random_state=random_state)
rf.fit(train['features'], train['target'].ravel())

prediction = rf.predict(test['features'])
test['predictions_rf'] = pd.DataFrame(prediction)


plt.rcParams["figure.figsize"] = [10.50, 5.50]
plt.rcParams["figure.autolayout"] = True

plt.subplot(121)
plt.title('Observations vs Predictions for RF model')
plt.xlabel("Q measured m3/s/m"), plt.ylabel("Q predictions RF m3/s/m")
plt.scatter(transform(Hm0_test, test['target'],scaler),
            transform(Hm0_test, test['predictions_rf'],scaler),
            s = 10,marker = '.',c='red',
           )
plt.plot(np.linspace(0, 1),
         np.linspace(0, 1), ':k')
#plt.axis([0.00000001, 1, 0.00000001, 1])
plt.loglog()
plt.grid()

plt.subplot(122)
plt.title('Observations vs Predictions for ANN model')
plt.xlabel("Q measured m3/s/m"), plt.ylabel("Q predictions ANN m3/s/m")
plt.scatter(ANN_1_q['q'],
            nn_overtoppingANN_1['Average'],
            s = 10,marker = '.',c='blue',
           )
plt.plot(np.linspace(0, 1),
         np.linspace(0, 1), ':k')
#plt.axis([0.00000001, 1, 0.00000001, 1])
plt.loglog()
plt.grid()


###########################################

ANN_1_q = ANN_1_q.reset_index(drop=True)
nn_overtoppingANN_1 = nn_overtoppingANN_1.reset_index(drop=True)

print("############## SCORE DATASET TEST FOR ANN ########")
print()
print("R2 TEST")
print(r2_score(ANN_1_q['q'], nn_overtoppingANN_1['Average']))
print()
print("RMSE TEST")
print(np.sqrt(mean_squared_error(ANN_1_q['q'], nn_overtoppingANN_1['Average']))) 


fig = plt.figure(figsize=(20, 30))
ax1 = fig.add_subplot(421)
plt.title('Observations vs Predictions for ANN model', fontsize=24)
plt.xlabel("Q measured m3/s/m", fontsize=18), plt.ylabel("Q predictions ANN m3/s/m", fontsize=18)

plt.scatter(ANN_1_q['q'],
            nn_overtoppingANN_1['Average'],
            s = 10,marker = '.',c='red',
           )
plt.plot(np.linspace(0, 1),
         np.linspace(0, 1), ':k')
#plt.axis([0.00000001, 1, 0.00000001, 1])
plt.loglog()
plt.grid()