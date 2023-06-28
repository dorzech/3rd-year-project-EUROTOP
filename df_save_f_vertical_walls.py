# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 19:25:45 2023

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
df1= pd.read_excel('Database_Eurotop_v5.xlsx')
df= pd.read_excel('Database_Eurotop_v5.xlsx')

label = pd.DataFrame(df['Label'])
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

df = pd.DataFrame({
    #'Label':df['Label'],
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
#for x in delete_list:
#    df = df.drop(columns=[x])


#delete q == 0
df.drop(df.index[df['q'] == 0 ], inplace=True)

#df.drop(df.index[(df['Hm0']/['Lm0']) < 0.01 ], inplace=True)
#df.drop(df.index[(df['Hm0']/['Lm0']) > 0.06 ], inplace=True)

df = df.astype('float')

df = df.drop(columns =['Kr', 'Kt'])
#df = df.drop(columns =['D', 'gf', 'cotaincl', 'Kr', 'Kt'])
df.insert(loc=0, column='Label', value=label)



# convert in float
#dfsave.drop(dfsave.head(1).index, inplace=True)

a = df.describe()

true_datasets_discharge = pd.DataFrame(df['q'])

df = df.dropna()


#Obtain F vertical walls
df_f = df[~df['Label'].astype(str).str.startswith('A')]
df_f = df_f[~df_f['Label'].astype(str).str.startswith('B')]
df_f = df_f[~df_f['Label'].astype(str).str.startswith('C')]
df_f = df_f[~df_f['Label'].astype(str).str.startswith('D')]
df_f = df_f[~df_f['Label'].astype(str).str.startswith('E')]
df_f = df_f[~df_f['Label'].astype(str).str.startswith('G')]
true_dataset_F_discharge = pd.DataFrame(df_f['q'])



#Obtain A
df_a = df[~df['Label'].astype(str).str.startswith('F')]
df_a = df_a[~df_a['Label'].astype(str).str.startswith('B')]
df_a = df_a[~df_a['Label'].astype(str).str.startswith('C')]
df_a = df_a[~df_a['Label'].astype(str).str.startswith('D')]
df_a = df_a[~df_a['Label'].astype(str).str.startswith('E')]
df_a = df_a[~df_a['Label'].astype(str).str.startswith('G')]
true_dataset_A_discharge = pd.DataFrame(df_a['q'])


#Obtain B
df_b = df[~df['Label'].astype(str).str.startswith('A')]
df_b = df_b[~df_b['Label'].astype(str).str.startswith('F')]
df_b = df_b[~df_b['Label'].astype(str).str.startswith('C')]
df_b = df_b[~df_b['Label'].astype(str).str.startswith('D')]
df_b = df_b[~df_b['Label'].astype(str).str.startswith('E')]
df_b = df_b[~df_b['Label'].astype(str).str.startswith('G')]
true_dataset_B_discharge = pd.DataFrame(df_b['q'])
 

#Obtain C
df_c = df[~df['Label'].astype(str).str.startswith('A')]
df_c = df_c[~df_c['Label'].astype(str).str.startswith('F')]
df_c = df_c[~df_c['Label'].astype(str).str.startswith('B')]
df_c = df_c[~df_c['Label'].astype(str).str.startswith('D')]
df_c = df_c[~df_c['Label'].astype(str).str.startswith('E')]
df_c = df_c[~df_c['Label'].astype(str).str.startswith('G')]
true_dataset_C_discharge = pd.DataFrame(df_c['q'])


#Obtain D
df_d = df[~df['Label'].astype(str).str.startswith('A')]
df_d = df_d[~df_d['Label'].astype(str).str.startswith('F')]
df_d = df_d[~df_d['Label'].astype(str).str.startswith('B')]
df_d = df_d[~df_d['Label'].astype(str).str.startswith('C')]
df_d = df_d[~df_d['Label'].astype(str).str.startswith('E')]
df_d = df_d[~df_d['Label'].astype(str).str.startswith('G')]
true_dataset_D_discharge = pd.DataFrame(df_d['q'])


#Obtain E
df_e = df[~df['Label'].astype(str).str.startswith('A')]
df_e = df_e[~df_e['Label'].astype(str).str.startswith('F')]
df_e = df_e[~df_e['Label'].astype(str).str.startswith('B')]
df_e = df_e[~df_e['Label'].astype(str).str.startswith('C')]
df_e = df_e[~df_e['Label'].astype(str).str.startswith('D')]
df_e = df_e[~df_e['Label'].astype(str).str.startswith('G')]
true_dataset_e_discharge = pd.DataFrame(df_e['q'])


#Obtain G
df_g = df[~df['Label'].astype(str).str.startswith('A')]
df_g = df_g[~df_g['Label'].astype(str).str.startswith('F')]
df_g = df_g[~df_g['Label'].astype(str).str.startswith('B')]
df_g = df_g[~df_g['Label'].astype(str).str.startswith('C')]
df_g = df_g[~df_g['Label'].astype(str).str.startswith('D')]
df_g = df_g[~df_g['Label'].astype(str).str.startswith('E')]
true_dataset_g_discharge = pd.DataFrame(df_g['q'])

df = df.drop(columns =['q'])

#prepare file for ANN tool
df['Kr'] = 0
df['Kt'] = 0
df['q'] = 1

#Obtain A
df_a = df[~df['Label'].astype(str).str.startswith('F')]
df_a = df_a[~df_a['Label'].astype(str).str.startswith('B')]
df_a = df_a[~df_a['Label'].astype(str).str.startswith('C')]
df_a = df_a[~df_a['Label'].astype(str).str.startswith('D')]
df_a = df_a[~df_a['Label'].astype(str).str.startswith('E')]
df_a = df_a[~df_a['Label'].astype(str).str.startswith('G')]

df_a.to_csv('ANN_OVERTOPPING4_14_03_DF_A_1.csv', decimal=".", sep='|', index = False ) 

#Obtain B
df_b = df[~df['Label'].astype(str).str.startswith('A')]
df_b = df_b[~df_b['Label'].astype(str).str.startswith('F')]
df_b = df_b[~df_b['Label'].astype(str).str.startswith('C')]
df_b = df_b[~df_b['Label'].astype(str).str.startswith('D')]
df_b = df_b[~df_b['Label'].astype(str).str.startswith('E')]
df_b = df_b[~df_b['Label'].astype(str).str.startswith('G')]

df_b.to_csv('ANN_OVERTOPPING4_14_03_DF_B_1.csv', decimal=".", sep='|', index = False ) 

#Obtain C
df_c = df[~df['Label'].astype(str).str.startswith('A')]
df_c = df_c[~df_c['Label'].astype(str).str.startswith('F')]
df_c = df_c[~df_c['Label'].astype(str).str.startswith('B')]
df_c = df_c[~df_c['Label'].astype(str).str.startswith('D')]
df_c = df_c[~df_c['Label'].astype(str).str.startswith('E')]
df_c = df_c[~df_c['Label'].astype(str).str.startswith('G')]

df_c.to_csv('ANN_OVERTOPPING4_14_03_DF_C_1.csv', decimal=".", sep='|', index = False ) 

#Obtain D
df_d = df[~df['Label'].astype(str).str.startswith('A')]
df_d = df_d[~df_d['Label'].astype(str).str.startswith('F')]
df_d = df_d[~df_d['Label'].astype(str).str.startswith('B')]
df_d = df_d[~df_d['Label'].astype(str).str.startswith('C')]
df_d = df_d[~df_d['Label'].astype(str).str.startswith('E')]
df_d = df_d[~df_d['Label'].astype(str).str.startswith('G')]

df_d.to_csv('ANN_OVERTOPPING4_14_03_DF_D_1.csv', decimal=".", sep='|', index = False ) 

#Obtain E
df_e = df[~df['Label'].astype(str).str.startswith('A')]
df_e = df_e[~df_e['Label'].astype(str).str.startswith('F')]
df_e = df_e[~df_e['Label'].astype(str).str.startswith('B')]
df_e = df_e[~df_e['Label'].astype(str).str.startswith('C')]
df_e = df_e[~df_e['Label'].astype(str).str.startswith('D')]
df_e = df_e[~df_e['Label'].astype(str).str.startswith('G')]

df_e.to_csv('ANN_OVERTOPPING4_14_03_DF_E_1.csv', decimal=".", sep='|', index = False ) 

#Obtain G
df_g = df[~df['Label'].astype(str).str.startswith('A')]
df_g = df_g[~df_g['Label'].astype(str).str.startswith('F')]
df_g = df_g[~df_g['Label'].astype(str).str.startswith('B')]
df_g = df_g[~df_g['Label'].astype(str).str.startswith('C')]
df_g = df_g[~df_g['Label'].astype(str).str.startswith('D')]
df_g = df_g[~df_g['Label'].astype(str).str.startswith('E')]

df_g.to_csv('ANN_OVERTOPPING4_14_03_DF_G_1.csv', decimal=".", sep='|', index = False ) 

#Obtain F
df_f = df[~df['Label'].astype(str).str.startswith('A')]
df_f = df_f[~df_f['Label'].astype(str).str.startswith('G')]
df_f = df_f[~df_f['Label'].astype(str).str.startswith('B')]
df_f = df_f[~df_f['Label'].astype(str).str.startswith('C')]
df_f = df_f[~df_f['Label'].astype(str).str.startswith('D')]
df_f = df_f[~df_f['Label'].astype(str).str.startswith('E')]

df_f.to_csv('ANN_OVERTOPPING4_14_03_DF_F_1.csv', decimal=".", sep='|', index = False ) 

#############################################################################################
df_f['Lm0'] = ((9.81*(df_f['Tm1,0t'])**2)/(2*np.pi))
# create dataframe containing the model parameters
df_f_save = pd.DataFrame({
    'Hm0' :df_f['Hm0 toe'],
    'Hm0/Lm0':df_f['Hm0 toe']/df_f['Lm0'],
    'β':df_f['β'],
    'h/Lm0':df_f['h']/df_f['Lm0'],
    'ht/Hm0':df_f['ht']/df_f['Hm0 toe'],
    'Bt/Lm0':df_f['Bt']/df_f['Lm0'],
    'hb/Hm0':df_f['hb']/df_f['Hm0 toe'],
    'B/Lm0':df_f['B']/df_f['Lm0'],
    'Ac/Hm0':df_f['Ac']/df_f['Hm0 toe'],
    'Rc/Hm0':df_f['Rc']/df_f['Hm0 toe'],
    #'Rc':df_f['Rc'],
    'Gc/Lm0':df_f['Gc']/df_f['Lm0'],
    'm':df_f['m'],
    'cotad':df_f['cotad'],
    'cotaincl':df_f['cotaincl'],
    'gf':df_f['gf'],
    'q':df_f['q'],
    'D/Hm0':df_f['D']/df_f['Hm0 toe'],
    #target:
    'qAD':np.log10(df_f['q']/np.sqrt(9.81*(df_f['Hm0 toe'])**3)),
    })
#head_f = pd.DataFrame(df_f.head())
df_f_describe = pd.DataFrame(df_f_save.describe())
range_f = pd.DataFrame(df_f_describe.iloc[[3]].T)
range_f['mean']= pd.DataFrame(df_f_describe.iloc[[1]].T)
range_f['max'] = pd.DataFrame(df_f_describe.iloc[[7]].T)



discharge_dfsave_output = df.drop(columns =['D', 'gf', 'cotaincl'])
#discharge_dfsave_output.insert(loc=0, column='Label', value=df['Label'])

#delete rows with Nan values of'q'

#print(dfsave_output.dtypes)
#dfsave_output.head()


#delete 'Label' to change data type for float
#for x in delete_list:
#   dfsave = dfsave.drop(columns=[x])

#c= df_f_save.describe()
# Replace infinite updated data with nan
df_f_save.replace([np.inf, -np.inf], np.nan, inplace=True)
# Drop rows with NaN
df_f_save.dropna(inplace=True)

#dfsave.to_csv('NN_OVERTOPPING6.csv', decimal=".", sep=' ', header = False, index = False ) 
#val_=dfsave.describe()

###############3
df_f['Lm0'] = ((9.81*(df_f['Tm1,0t'])**2)/(2*np.pi))
df_f_randomsearch = pd.DataFrame({
    'Hm0/Lm0':df_f_save['Hm0/Lm0'],
    'β':df_f_save['β'],
    'h/Lm0':df_f_save['h/Lm0'],
    'ht/Hm0':df_f_save['ht/Hm0'],
    'Bt/Lm0':df_f_save['Bt/Lm0'],
    'hb/Hm0':df_f_save['hb/Hm0'],
    'B/Lm0':df_f_save['B/Lm0'],
    'Ac/Hm0':df_f_save['Ac/Hm0'],
    'Rc/Hm0':df_f_save['Rc/Hm0'],
    #'Rc':df_f['Rc'],
    'Gc/Lm0':df_f_save['Gc/Lm0'],
    'm':df_f_save['m'],
    'cotad':df_f_save['cotad'],
    'cotaincl':df_f_save['cotaincl'],
    'gf':df_f_save['gf'],
    
    'D/Hm0':df_f_save['D/Hm0'],
    })

df_f_randomsearch.replace([np.inf, -np.inf], np.nan, inplace=True)
# Drop rows with NaN
df_f_randomsearch.dropna(inplace=True)

df_f_randomsearch_output = pd.DataFrame({ 'qAD':df_f_save['qAD']})

df_f_randomsearch_output.replace([np.inf, -np.inf], np.nan, inplace=True)
# Drop rows with NaN
df_f_randomsearch_output.dropna(inplace=True)



##########################################################################################


train_set, test_set = train_test_split(df_f_save, test_size=0.2,random_state=1123)


scaler = StandardScaler()
# transform data

#scaler_std = StandardScaler()
train_transform = pd.DataFrame(train_set.drop(['Hm0','q','qAD'], axis=1))
#desc_train_transform = train_transform.describe()
#desc_train_transform.to_excel('desc_train_transform.xlsx') 
train_transform = pd.DataFrame(scaler.fit_transform(train_transform))
train_transform.columns = train_set.drop(['Hm0','q','qAD'], axis=1).columns

test_transform = pd.DataFrame(test_set.drop(['Hm0','q','qAD'], axis=1))
#desc_test_transform = test_transform.describe()
#desc_test_transform.to_csv('desc_test_transform.csv', decimal=".", sep=' ', header = True, index = False ) 
test_transform = pd.DataFrame(scaler.transform(test_transform))
test_transform.columns = test_set.drop(['Hm0','q','qAD'], axis=1).columns


#for i in train_transform.columns:
#    fig = plt.hist(train_transform[i],bins=20)3
#    plt.title('Transformed '+i)
#    plt.show(fig)

#organize the data 
train = {
    'features': train_transform,
    'target' : scaler.fit_transform(train_set['qAD'].values.reshape(-1,1))
}
test = {
    'features': test_transform,
    'target' : pd.DataFrame(scaler.fit_transform(test_set['qAD'].values.reshape(-1,1)))
}

# keep Hm0 data for the inverse_transform
Hm0_test = pd.DataFrame(test_set['Hm0'])


    # select the metrics
scoring = ['neg_mean_squared_error','r2' ]


#
def transform(Hm0, data,scaler):
    prediction = scaler.inverse_transform(data.values.reshape(-1,1))
    q_overtopping = (10**(prediction.flatten()))*np.sqrt(9.81*(Hm0_test['Hm0'])**3)
    return q_overtopping



#from sklearn.model_selection import GridSearchCV
# RF Regressor

### Add validation data set
# split the train set between train and validation set
train_feature_rf, train_val_feature_rf, train_target_rf, train_val_target_rf = train_test_split(train['features'], train['target'], test_size=0.2,random_state=1123)
train['features_rf'] = train_feature_rf
train['target_rf'] = train_target_rf
train['val_feature_rf'] = train_val_feature_rf
train['val_target_rf'] = train_val_target_rf


#desc_train_feature_rf = pd.DataFrame(train_feature_rf).describe()
#desc_train_target_rf = pd.DataFrame(train_target_rf).describe()
#desc_train_val_feature_rf = train_val_feature_rf.describe()
#desc_train_val_target_rf = pd.DataFrame(train_val_target_rf).describe()

# build function for hyperparameters search

def RandomForestR(params, cv, scoring, features, target):

    rf = RandomForestRegressor(random_state = 1123)
   
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
    'max_depth': [24, 34, 48],
    #'max_features': ['auto', 0.25],
    'min_samples_leaf': [2, 5],
    #'min_samples_split': [6, 32],
    'n_estimators': [116, 141]
}



RandomForestR(params_grid, 5, scoring, train['features'], train['target'].ravel())




### Search the best n_estimator
# search best n_estimators


#sum_of_errors=[]
#for i in range(1, 51):
#    print(i)
rf = RandomForestRegressor( max_depth=34, min_samples_leaf = 2, n_estimators=356)
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
bst_n_estimators = 356

rf_best = RandomForestRegressor(max_depth=34, n_estimators=bst_n_estimators, random_state=1123)
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
            importances[indices], color="blue", align="center")
    plt.xticks(range(len(importances)),
               features.columns[indices], fontsize=10, rotation=90)
    plt.xlim([-1, len(importances)])
    plt.show()
    
features_importances(rf,train['features'])
### Build the rf model and make predictions
# run rf with best parameters
# add stochastic rf => reduce Std but increase biais.
rf = RandomForestRegressor(     max_depth=34,
                                n_estimators=356,
                                random_state=1123)
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

plt.rcParams["figure.figsize"] = [10.50, 7.50]
plt.rcParams["figure.autolayout"] = True
plt.title('Observations vs Predictions for rf model')
plt.xlabel("Q measured m3/s/m"), plt.ylabel("Q predictions rf m3/s/m")
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
rf = RandomForestRegressor(     max_depth=26,
                                n_estimators=133,
                                random_state=1123)
rf.fit(train['features'], train['target'].ravel())

prediction = rf.predict(test['features'])
test['predictions_rf'] = pd.DataFrame(prediction)


plt.rcParams["figure.figsize"] = [10.50, 5.50]
plt.rcParams["figure.autolayout"] = True

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
plt.title('Observations vs Predictions for ANN model for vertical walls')
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