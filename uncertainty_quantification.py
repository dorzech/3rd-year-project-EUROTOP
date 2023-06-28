# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 19:49:44 2022

@author: to ja
"""


# Import libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.ensemble import VotingRegressor

from sklearn.ensemble import RandomForestRegressor

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Load the data 
df= pd.read_excel('Database_Eurotop_v5.xlsx')

output_F = pd.read_csv('ANN_OVERTOPPING4_13_03_F.csv', decimal=".", sep='|') 

#dfx= pd.read_excel('Database_Eurotop_v5.xlsx')
df.head()

#create delete list for parameters unused
delete_list = ['Label']

#delete non-core data
df.drop(df.index[df['Core data'] != 'Z'], inplace=True)

#delete rows with Nan values of'q'
df = df.dropna(subset=['q'])

df_f = df[~df['Label'].astype(str).str.startswith('A')]
df_f = df_f[~df_f['Label'].astype(str).str.startswith('B')]
df_f = df_f[~df_f['Label'].astype(str).str.startswith('C')]
df_f = df_f[~df_f['Label'].astype(str).str.startswith('D')]
df_f = df_f[~df_f['Label'].astype(str).str.startswith('E')]
df_f = df_f[~df_f['Label'].astype(str).str.startswith('G')]
df_f.to_csv('ANN_OVERTOPPING4_13_03_DF_F_1.csv', decimal=".", sep='|', index = False ) 


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


#check the data types
#print(dfsave.dtypes)

#delete rows for 'q' = 0
#t = pd.DataFrame((dfsave.index[dfsave['q'] != 0]))
t = pd.DataFrame(dfsave.drop(dfsave.index[dfsave['q'] != 0]))

#delete q == 0
dfsave.drop(dfsave.index[dfsave['q'] == 0 ], inplace=True)

#delete 'Label' to change data type for float
for x in delete_list:
    dfsave = dfsave.drop(columns=[x])

# convert in float
#dfsave.drop(dfsave.head(1).index, inplace=True)
dfsave = dfsave.astype('float')
dfsave = dfsave.dropna(subset=['m'])
dfsave = dfsave.dropna(subset=['D'])
a = dfsave.describe()

#a.to_csv('A1.csv', decimal=",", sep=' ', index = False )


# delete the first line (units)
#df.drop(df.head(1).index, inplace=True)

#%matplotlib qt
####################################################
#for i in dfsave.columns:
#    fig = plt.hist(dfsave[i],bins=10)
#    plt.title(i)
#    plt.show(fig)

# delete the NaN 
#dfsave = dfsave.dropna()
#a = df.describe()

##################corr matrix to see correlation of parameters
#corr_matrix = dfsave.corr()
#f, ax = plt.subplots(figsize=(20,20))
#g = sns.heatmap(corr_matrix, cmap = sns.diverging_palette(10, 250, as_cmap = True), annot = True)

#creating an additional feature - compute spectral wave length in deep water lm0 in meter
dfsave['Lm0'] = ((9.81*(dfsave['Tm1,0t'])**2)/(2*np.pi))

# create dataframe containing the model parameters
dff = pd.DataFrame({
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
#a=dfsave.describe()


#a = pd.DataFrame(dff.index[dff['hb/Hm0'] < -2.133 ])
#b = pd.DataFrame(dff.index[dff['hb/Hm0'] > 7.143 ])
#c = pd.DataFrame(dff.index[dff['Gc/Lm0'] > 0.362 ])
#d = pd.DataFrame(dff.index[dff['B/Lm0'] > 0.972 ])
#e = pd.DataFrame(dff.index[dff['h/Lm0'] < 0.003  ])
#f = pd.DataFrame(dff.index[dff['Hm0/Lm0'] < 0.002  ])

##delete out of range of validity
#dfsave.drop(dff.index[dff['hb/Hm0'] < -2.133 ], inplace=True)
#dfsave.drop(dff.index[dff['hb/Hm0'] > 7.143 ], inplace=True)
#dfsave.drop(dff.index[dff['Gc/Lm0'] > 0.362 ], inplace=True)
#dfsave.drop(dff.index[dff['B/Lm0'] > 0.972 ], inplace=True)
#dfsave.drop(dff.index[dff['h/Lm0'] < 0.003  ], inplace=True)
#dfsave.drop(dff.index[dff['Hm0/Lm0'] < 0.002  ], inplace=True)



discharge_dfsave_output = dfsave.drop(columns =['D', 'gf', 'cotaincl', 'Lm0', 'Kr', 'Kt'])
discharge_dfsave_output.insert(loc=0, column='Label', value=df['Label'])

discharge_dfsave_output_F = discharge_dfsave_output[~discharge_dfsave_output['Label'].astype(str).str.startswith('A')]
discharge_dfsave_output_F = discharge_dfsave_output_F[~discharge_dfsave_output_F['Label'].astype(str).str.startswith('B')]
discharge_dfsave_output_F = discharge_dfsave_output_F[~discharge_dfsave_output_F['Label'].astype(str).str.startswith('C')]
discharge_dfsave_output_F = discharge_dfsave_output_F[~discharge_dfsave_output_F['Label'].astype(str).str.startswith('D')]
discharge_dfsave_output_F = discharge_dfsave_output_F[~discharge_dfsave_output_F['Label'].astype(str).str.startswith('E')]
discharge_dfsave_output_F = discharge_dfsave_output_F[~discharge_dfsave_output_F['Label'].astype(str).str.startswith('G')]






dfsave_output = dfsave.drop(columns =['D', 'gf', 'cotaincl', 'Lm0','q', 'Kr', 'Kt'])

f_discharge = pd.DataFrame(dfsave['q'])

dfsave_output = dfsave_output.dropna()

#delete rows with Nan values of'q'

#print(dfsave_output.dtypes)
#dfsave_output.head()

#prepare file for ANN tool
dfsave_output['Kr'] = 0
dfsave_output['Kt'] = 0
dfsave_output['q'] = 1


dfsave_output.insert(loc=0, column='Label', value=df['Label'])
#dfsave_output.drop(dfsave.head(1).index, inplace=True)

dfsave_output_F = dfsave_output[~dfsave_output['Label'].astype(str).str.startswith('A')]
dfsave_output_F = dfsave_output_F[~dfsave_output_F['Label'].astype(str).str.startswith('B')]
dfsave_output_F = dfsave_output_F[~dfsave_output_F['Label'].astype(str).str.startswith('C')]
dfsave_output_F = dfsave_output_F[~dfsave_output_F['Label'].astype(str).str.startswith('D')]
dfsave_output_F = dfsave_output_F[~dfsave_output_F['Label'].astype(str).str.startswith('E')]
dfsave_output_F = dfsave_output_F[~dfsave_output_F['Label'].astype(str).str.startswith('G')]

dfsave_output_F.to_csv('ANN_OVERTOPPING4_13_03_F.csv', decimal=".", sep='|', index = False ) 
dfsave_output.to_csv('ANN_OVERTOPPING4_13_03_B.csv', decimal=".", sep='|', index = False ) 
  

#delete 'Label' to change data type for float
#for x in delete_list:
#   dfsave = dfsave.drop(columns=[x])

#c= dff.describe()
# Replace infinite updated data with nan
dff.replace([np.inf, -np.inf], np.nan, inplace=True)
# Drop rows with NaN
dff.dropna(inplace=True)

#dfsave.to_csv('NN_OVERTOPPING6.csv', decimal=".", sep=' ', header = False, index = False ) 
#val_=dfsave.describe()

##################################
#1. important feature Rc/Hm0
#2. important feature D/Hm0
#3. important feature Rc/Hm0
##################################


train_set, test_set = train_test_split(dff, test_size=0.2,random_state=42)

test_set_describe = test_set.describe()

mean = 0
var = 0.1
sigma = var**0.8
gauss = np.random.normal(mean, sigma, (row,col,ch))
gauss = gauss.reshape(row, col, ch)


scaler = StandardScaler()
# transform data

#scaler_std = StandardScaler()
train_transform = pd.DataFrame(train_set.drop(['Hm0','q','qAD'], axis=1))
train_transform = pd.DataFrame(scaler.fit_transform(train_transform))
train_transform.columns = train_set.drop(['Hm0','q','qAD'], axis=1).columns
test_transform = pd.DataFrame(test_set.drop(['Hm0','q','qAD'], axis=1))
test_transform = pd.DataFrame(scaler.transform(test_transform))
test_transform.columns = test_set.drop(['Hm0','q','qAD'], axis=1).columns


for i in train_transform.columns:
    fig = plt.hist(train_transform[i],bins=20)
    plt.title('Transformed '+i)
    plt.show(fig)

#organize the data 
train = {
    #'features': train_transform.drop(['Hm0','q','qAD','qAD'], axis=1),
    #'target' : train_transform['qAD']
    #'features': train_set.drop(['Hm0','q','qAD'], axis=1),
    'features': train_transform,
    'target' : scaler.fit_transform(train_set['qAD'].values.reshape(-1,1))
}
test = {
    #'features': test_transform.drop(['Hm0','q','qAD','qAD'], axis=1),
    #'target' : test_transform['qAD']
    #'features': test_set.drop(['Hm0','q','qAD'], axis=1),
    'features': test_transform,
    'target' : pd.DataFrame(scaler.fit_transform(test_set['qAD'].values.reshape(-1,1)))
}

# keep Hm0 data for the inverse_transform
Hm0_test = pd.DataFrame(test_set['Hm0'])


    # select the metrics
scoring = ['neg_mean_squared_error','r2']


# plot the SVR predictions vs Observation from the test set
#
#
#
def transform(Hm0, data,scaler):
    prediction = scaler.inverse_transform(data.values.reshape(-1,1))
    q_overtopping = (10**(prediction.flatten()))*np.sqrt(9.81*(Hm0_test['Hm0'])**3)
    return q_overtopping



#from sklearn.model_selection import GridSearchCV
# RF Regressor

### Add validation data set
# split the train set between train and validation set
train_feature_rf, train_val_feature_rf, train_target_rf, train_val_target_rf = train_test_split(train['features'], train['target'], test_size=0.2,random_state=42)
train['features_rf'] = train_feature_rf
train['target_rf'] = train_target_rf
train['val_feature_rf'] = train_val_feature_rf
train['val_target_rf'] = train_val_target_rf
### Hyperparameters search
# build function for hyperparameters search

def RandomForestR(params,  scoring, features, target):

    rf = RandomForestRegressor(random_state = 42)
   
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
    'max_depth': [15, 20, 42],
    #'max_features': ['auto'],
    'min_samples_leaf': [10, 24],
    #'min_samples_split': [6, 32],
    'n_estimators': [124, 150, 460]
}



RandomForestR(params_grid, scoring, train['features'], train['target'].ravel())

### Search the best n_estimator
# search best n_estimators


#sum_of_errors=[]
#for i in range(1, 51):
#    print(i)
   rf = RandomForestRegressor( max_depth=20, n_estimators=460)
   rf.fit(train['features_rf'],train['target_rf'])

    #errors = [mean_squared_error(train['val_target_rf'], prediction)
    #     for prediction in rf.predict(train['val_feature_rf'])]
#    errors = []
#    for real, pred in zip(train['val_target_rf'], rf.predict(train['val_feature_rf'])):
#        try:
#            errors.append(mean_squared_error(real, [pred]))    
#        except:
#            print(real, [pred])

#    sum_of_errors.append(sum(errors))

#bst_n_estimators = np.argmin(errors) + 1
bst_n_estimators = 460

rf_best = RandomForestRegressor(max_depth=20, n_estimators=bst_n_estimators, random_state=42)
rf_best.fit(train['features_rf'], train['target_rf'])

#plot validation error for the n_estimators
#min_error = np.min(errors)
#plt.figure(figsize=(20, 10))
#plt.plot(np.arange(1, len(errors) + 1), errors, "b.-")
#plt.plot([bst_n_estimators, bst_n_estimators], [0, min_error], "k--")
#plt.plot([0, 820], [min_error, min_error], "k--")
#plt.plot(bst_n_estimators, min_error, "ko")
#plt.text(bst_n_estimators, min_error*1.2, "Minimum", ha="center", fontsize=14)
#plt.axis([0, 800, 0, 1])
#plt.xlabel("Number of trees")
#plt.ylabel("Error", fontsize=16)
#plt.title("Validation error", fontsize=14)

###  Feature importances from t model
# check the feature importance

def features_importances(estimator, features):
    importances = estimator.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(15, 10))
    plt.title("Feature importances")
    plt.bar(range(len(importances)),
            importances[indices], color="#dddddd", align="center")
    plt.xticks(range(len(importances)),
               features.columns[indices], fontsize=10, rotation=90)
    plt.xlim([-1, len(importances)])
    plt.show()
    
features_importances(rf,train['features'])
### Build the rf model and make predictions
# run rf with best parameters
# add stochastic rf => reduce Std but increase biais.
rf = RandomForestRegressor(     max_depth=11,
                                n_estimators=487,
                                random_state=42)
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
def transform(Hm0, data,scaler):
    prediction = scaler.inverse_transform(data.values.reshape(-1,1))
    q_overtopping = (10**(prediction.flatten()))*np.sqrt(9.81*(Hm0_test['Hm0'])**3)
    return q_overtopping

fig = plt.figure(figsize=(15,10))
plt.title('Observations vs Predictions for rf model')
plt.xlabel("Q measured m3/s/m"), plt.ylabel("Q predictions rf m3/s/m")
plt.scatter(transform(Hm0_test, test['target'],scaler),
            transform(Hm0_test, test['predictions'],scaler),
            s = 10,marker = '.',c='gray',
           )
plt.plot(np.linspace(0, 1),
         np.linspace(0, 1), ':k')
#plt.axis([0.00000001, 1, 0.00000001, 1])
plt.loglog()
plt.grid()
# rf model vs SVR model



#Compare the score and predictions from theses models
rf = RandomForestRegressor(     max_depth=20,
                                n_estimators=460,
                                random_state=42)
rf.fit(train['features'], train['target'].ravel())

prediction = rf.predict(test['features'])
test['predictions_rf'] = pd.DataFrame(prediction)



plt.subplot(121)
plt.title('Observations vs Predictions for rf model')
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
plt.scatter(discharge_dfsave_output_F['q'],
            output_F['Average'],
            s = 10,marker = '.',c='blue',
           )
plt.plot(np.linspace(0, 1),
         np.linspace(0, 1), ':k')
#plt.axis([0.00000001, 1, 0.00000001, 1])
plt.loglog()
plt.grid()

