# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 19:49:44 2022

@author: to ja
"""


# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

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

################################################################################################

nn_overtoppingA_q = pd.read_csv(r"C:\Users\to ja\.spyder-py3\eurotop_data\nn_overtoppingA_q.csv", decimal=".", sep='|')
nn_overtoppingB_q = pd.read_csv(r"C:\Users\to ja\.spyder-py3\eurotop_data\nn_overtoppingB_q.csv", decimal=".", sep='|')
nn_overtoppingC_q = pd.read_csv(r"C:\Users\to ja\.spyder-py3\eurotop_data\nn_overtoppingC_q.csv", decimal=".", sep='|')
nn_overtoppingD_q = pd.read_csv(r"C:\Users\to ja\.spyder-py3\eurotop_data\nn_overtoppingD_q.csv", decimal=".", sep='|')
nn_overtoppingE_q = pd.read_csv(r"C:\Users\to ja\.spyder-py3\eurotop_data\nn_overtoppingE_q.csv", decimal=".", sep='|')
nn_overtoppingF_q = pd.read_csv(r"C:\Users\to ja\.spyder-py3\eurotop_data\nn_overtoppingF_q.csv", decimal=".", sep='|')
nn_overtoppingG_q = pd.read_csv(r"C:\Users\to ja\.spyder-py3\eurotop_data\nn_overtoppingG_q.csv", decimal=".", sep='|')


fig = plt.figure(figsize=(20, 30))
ax1 = fig.add_subplot(421)
plt.title('Observations vs Predictions for ANN model for A', fontsize=24)
plt.xlabel("Q measured m3/s/m", fontsize=18), plt.ylabel("Q predictions ANN m3/s/m", fontsize=18)

plt.scatter(true_dataset_A_discharge['q'],
            nn_overtoppingA_q['Average'],
            s = 10,marker = '.',c='red',
           )
plt.plot(np.linspace(0, 1),
         np.linspace(0, 1), ':k')
#plt.axis([0.00000001, 1, 0.00000001, 1])
plt.loglog()
plt.grid()

plt.subplot(422)
plt.title('Observations vs Predictions for ANN model for B', fontsize=24)
plt.xlabel("Q measured m3/s/m", fontsize=18), plt.ylabel("Q predictions ANN m3/s/m", fontsize=18)
plt.scatter(true_dataset_B_discharge['q'],
            nn_overtoppingB_q['Average'],
            s = 10,marker = '.',c='blue',
           )
plt.plot(np.linspace(0, 1),
         np.linspace(0, 1), ':k')
#plt.axis([0.00000001, 1, 0.00000001, 1])
plt.loglog()
plt.grid()


plt.subplot(423)
plt.title('Observations vs Predictions for ANN model for C', fontsize=24)
plt.xlabel("Q measured m3/s/m", fontsize=18), plt.ylabel("Q predictions ANN m3/s/m", fontsize=18)
plt.scatter(true_dataset_C_discharge['q'],
            nn_overtoppingC_q['Average'],
            s = 10,marker = '.',c='green',
           )
plt.plot(np.linspace(0, 1),
         np.linspace(0, 1), ':k')

#plt.axis([0.00000001, 1, 0.00000001, 1])
plt.loglog()
plt.grid()

plt.subplot(424)
plt.title('Observations vs Predictions for ANN model for D', fontsize=24)
plt.xlabel("Q measured m3/s/m", fontsize=18), plt.ylabel("Q predictions ANN m3/s/m", fontsize=18)
plt.scatter(true_dataset_D_discharge['q'],
            nn_overtoppingD_q['Average'],
            s = 10,marker = '.',c='orange',
           )
plt.plot(np.linspace(0, 1),
         np.linspace(0, 1), ':k')
#plt.axis([0.00000001, 1, 0.00000001, 1])
plt.loglog()
plt.grid()

plt.subplot(425)
plt.title('Observations vs Predictions for ANN model for E', fontsize=24)
plt.xlabel("Q measured m3/s/m", fontsize=18), plt.ylabel("Q predictions ANN m3/s/m", fontsize=18)
plt.scatter(true_dataset_e_discharge['q'],
            nn_overtoppingE_q['Average'],
            s = 10,marker = '.',c='purple',
           )
plt.plot(np.linspace(0, 1),
         np.linspace(0, 1), ':k')
#plt.axis([0.00000001, 1, 0.00000001, 1])
plt.loglog()
plt.grid()

plt.subplot(426)
plt.title('Observations vs Predictions for ANN model for F', fontsize=24)
plt.xlabel("Q measured m3/s/m", fontsize=18), plt.ylabel("Q predictions ANN m3/s/m", fontsize=18)
plt.scatter(true_dataset_F_discharge['q'],
            nn_overtoppingF_q['Average'],
            s = 10,marker = '.',c='black',
           )
plt.plot(np.linspace(0, 1),
         np.linspace(0, 1), ':k')
#plt.axis([0.00000001, 1, 0.00000001, 1])
plt.loglog()
plt.grid()

plt.subplot(427)
plt.title('Observations vs Predictions for ANN model for G', fontsize=24)
plt.xlabel("Q measured m3/s/m", fontsize=18), plt.ylabel("Q predictions ANN m3/s/m", fontsize=18)
plt.scatter(true_dataset_g_discharge['q'],
            nn_overtoppingG_q['Average'],
            s = 10,marker = '.',c='black',
           )
plt.plot(np.linspace(0, 1),
         np.linspace(0, 1), ':k')
#plt.axis([0.00000001, 1, 0.00000001, 1])
plt.loglog()
plt.grid()

plt.subplots_adjust(bottom=1,
                    top=2,
                    wspace=3.5,
                    hspace=3.5)
plt.show()





true_dataset_A_discharge = true_dataset_A_discharge.reset_index(drop = True)                         
nn_overtoppingA_q['true_dataset_discharge'] = (true_dataset_A_discharge['q'])

true_dataset_B_discharge = true_dataset_B_discharge.reset_index(drop = True)                         
nn_overtoppingB_q['true_dataset_discharge'] = (true_dataset_B_discharge['q'])

true_dataset_C_discharge = true_dataset_C_discharge.reset_index(drop = True)                         
nn_overtoppingC_q['true_dataset_discharge'] = (true_dataset_C_discharge['q'])

true_dataset_D_discharge = true_dataset_D_discharge.reset_index(drop = True)                         
nn_overtoppingD_q['true_dataset_discharge'] = (true_dataset_D_discharge['q'])

true_dataset_e_discharge = true_dataset_e_discharge.reset_index(drop = True)                         
nn_overtoppingE_q['true_dataset_discharge'] = (true_dataset_e_discharge['q'])

true_dataset_F_discharge = true_dataset_F_discharge.reset_index(drop = True)                         
nn_overtoppingF_q['true_dataset_discharge'] = (true_dataset_F_discharge['q'])

true_dataset_g_discharge = true_dataset_g_discharge.reset_index(drop = True)                         
nn_overtoppingG_q['true_dataset_discharge'] = (true_dataset_g_discharge['q'])

#### ANN performance when fitted to all of the data and predicting all of the data it has been trained on.
nn_overtoppingAq1 = pd.DataFrame(nn_overtoppingA_q['true_dataset_discharge'])

nn_overtoppingAq1 = nn_overtoppingAq1.append(nn_overtoppingB_q[['true_dataset_discharge']], ignore_index = True)
nn_overtoppingAq1 = nn_overtoppingAq1.append(nn_overtoppingC_q[['true_dataset_discharge']], ignore_index = True)
nn_overtoppingAq1 = nn_overtoppingAq1.append(nn_overtoppingD_q[['true_dataset_discharge']], ignore_index = True)
nn_overtoppingAq1 = nn_overtoppingAq1.append(nn_overtoppingE_q[['true_dataset_discharge']], ignore_index = True)
nn_overtoppingAq1 = nn_overtoppingAq1.append(nn_overtoppingF_q[['true_dataset_discharge']], ignore_index = True)
nn_overtoppingAq1 = nn_overtoppingAq1.append(nn_overtoppingG_q[['true_dataset_discharge']], ignore_index = True)

nn_overtoppingAq2 = pd.DataFrame(nn_overtoppingA_q['Average'])
nn_overtoppingAq2 = nn_overtoppingAq2.append(nn_overtoppingB_q[['Average']], ignore_index = True)
nn_overtoppingAq2 = nn_overtoppingAq2.append(nn_overtoppingC_q[['Average']], ignore_index = True)
nn_overtoppingAq2 = nn_overtoppingAq2.append(nn_overtoppingD_q[['Average']], ignore_index = True)
nn_overtoppingAq2 = nn_overtoppingAq2.append(nn_overtoppingE_q[['Average']], ignore_index = True)
nn_overtoppingAq2 = nn_overtoppingAq2.append(nn_overtoppingF_q[['Average']], ignore_index = True)
nn_overtoppingAq2 = nn_overtoppingAq2.append(nn_overtoppingG_q[['Average']], ignore_index = True)

nn_overtopping_comparison = pd.DataFrame(nn_overtoppingAq1['true_dataset_discharge'])
nn_overtopping_comparison['Average'] = nn_overtoppingAq2['Average']

rms_all = mean_squared_error(nn_overtopping_comparison['true_dataset_discharge'], nn_overtopping_comparison['Average'], squared=False)

r2 = r2_score(nn_overtopping_comparison['true_dataset_discharge'], nn_overtopping_comparison['Average'])
################################################################################

fig = plt.figure(figsize=(20,20))
ax1 = fig.add_subplot(111)
plt.title('Observations vs Predictions for ANN model', fontsize=24)
plt.xlabel("Q measured m3/s/m", fontsize=18), plt.ylabel("Q predictions ANN m3/s/m", fontsize=18)

ax1.scatter(true_dataset_A_discharge['q'],
            nn_overtoppingA_q['Average'],
            s =40,marker = '.',c='red',
           )
ax1.scatter(true_dataset_B_discharge['q'],
            nn_overtoppingB_q['Average'],
            s = 40,marker = '.',c='blue',
           )
ax1.scatter(true_dataset_C_discharge['q'],
            nn_overtoppingC_q['Average'],
            s = 40,marker = '.',c='green',
           )
ax1.scatter(true_dataset_D_discharge['q'],
            nn_overtoppingD_q['Average'],
            s = 40,marker = '.',c='orange',
           )
ax1.scatter(true_dataset_e_discharge['q'],
            nn_overtoppingE_q['Average'],
            s = 40,marker = '.',c='purple',
           )
ax1.scatter(true_dataset_F_discharge['q'],
            nn_overtoppingF_q['Average'],
            s = 40,marker = '.',c='black',
           )
ax1.scatter(true_dataset_g_discharge['q'],
            nn_overtoppingG_q['Average'],
            s = 40,marker = '.',c='grey',
           )

plt.plot(np.linspace(0, 1),
         np.linspace(0, 1), ':k')
#plt.axis([0.00000001, 1, 0.00000001, 1])
plt.loglog()
plt.grid()


##################################################################################

A_rock_permeable = pd.read_csv(r"C:\Users\to ja\.spyder-py3\eurotop_data\A_rock_permeable.csv", decimal=".", sep='|')
B_rock_impremeable = pd.read_csv(r"C:\Users\to ja\.spyder-py3\eurotop_data\B_rock_impremeable.csv", decimal=".", sep='|')
C_armour_units = pd.read_csv(r"C:\Users\to ja\.spyder-py3\eurotop_data\C_armour_units.csv", decimal=".", sep='|')
D_smooth_impremeable_dikes = pd.read_csv(r"C:\Users\to ja\.spyder-py3\eurotop_data\D_smooth_impremeable_dikes.csv", decimal=".", sep='|')
E_composite_slopes_and_berms = pd.read_csv(r"C:\Users\to ja\.spyder-py3\eurotop_data\E_composite_slopes_and_berms.csv", decimal=".", sep='|')
F_vertical_walls = pd.read_csv(r"C:\Users\to ja\.spyder-py3\eurotop_data\F_vertical_walls.csv", decimal=".", sep='|')

fig = plt.figure(figsize=(20, 30))
ax1 = fig.add_subplot(421)
plt.title('Rc/Hs | q/(gHs^3)^(1/2) for A', fontsize=24)
plt.xlabel("Rc/Hs", fontsize=18), plt.ylabel("q/(gHs^3)^(1/2)", fontsize=18)

plt.scatter(A_rock_permeable['Rc/Hs'],
            A_rock_permeable['q/(gHs^3)^(1/2)'],
            s = 10,marker = '.',c='red',
           )
#plt.axis([0.00000001, 1, 0.00000001, 1])
plt.loglog()
plt.grid()

plt.subplot(422)
plt.title('Rc/Hs | q/(gHs^3)^(1/2) for B', fontsize=24)
plt.xlabel("Rc/Hs", fontsize=18), plt.ylabel("q/(gHs^3)^(1/2)", fontsize=18)
plt.scatter(B_rock_impremeable['Rc/Hs'],
            B_rock_impremeable['q/(gHs^3)^(1/2)'],
            s = 10,marker = '.',c='blue',
           )
plt.loglog()
plt.grid()



plt.subplot(423)
plt.title('Rc/Hs | q/(gHs^3)^(1/2) for C', fontsize=24)
plt.xlabel("Rc/Hs", fontsize=18), plt.ylabel("q/(gHs^3)^(1/2)", fontsize=18)
plt.scatter(C_armour_units['Rc/Hs'],
            C_armour_units['q/(gHs^3)^(1/2)'],
            s = 10,marker = '.',c='green',
           )

plt.loglog()
plt.grid()

plt.subplot(424)
plt.title('Rc/Hs | q/(gHs^3)^(1/2) for D', fontsize=24)
plt.xlabel("Rc/Hs", fontsize=18), plt.ylabel("q/(gHs^3)^(1/2)", fontsize=18)
plt.scatter(D_smooth_impremeable_dikes['Rc/Hs'],
            D_smooth_impremeable_dikes['q/(gHs^3)^(1/2)'],
            s = 10,marker = '.',c='orange',
           )
plt.loglog()
plt.grid()

plt.subplot(425)
plt.title('Rc/Hs | q/(gHs^3)^(1/2) for E', fontsize=24)
plt.xlabel("Rc/Hs", fontsize=18), plt.ylabel("q/(gHs^3)^(1/2)", fontsize=18)
plt.scatter(E_composite_slopes_and_berms['Rc/Hs'],
            E_composite_slopes_and_berms['q/(gHs^3)^(1/2)'],
            s = 10,marker = '.',c='purple',
           )
plt.loglog()
plt.grid()

plt.subplot(426)
plt.title('Rc/Hs | q/(gHs^3)^(1/2) for F', fontsize=24)
plt.xlabel("Rc/Hs", fontsize=18), plt.ylabel("q/(gHs^3)^(1/2)", fontsize=18)
plt.scatter(F_vertical_walls['Rc/Hs'],
            F_vertical_walls['q/(gHs^3)^(1/2)'],
            s = 10,marker = '.',c='black',
           )
plt.loglog()
plt.grid()

plt.subplots_adjust(bottom=1,
                    top=2,
                    wspace=3.5,
                    hspace=3.5)
plt.show()


#############################################################################################################

nn_overtoppingA_q = pd.read_csv(r"C:\Users\to ja\.spyder-py3\eurotop_data\nn_overtoppingA_q.csv", decimal=".", sep='|')
nn_overtoppingB_q = pd.read_csv(r"C:\Users\to ja\.spyder-py3\eurotop_data\nn_overtoppingB_q.csv", decimal=".", sep='|')
nn_overtoppingC_q = pd.read_csv(r"C:\Users\to ja\.spyder-py3\eurotop_data\nn_overtoppingC_q.csv", decimal=".", sep='|')
nn_overtoppingD_q = pd.read_csv(r"C:\Users\to ja\.spyder-py3\eurotop_data\nn_overtoppingD_q.csv", decimal=".", sep='|')
nn_overtoppingE_q = pd.read_csv(r"C:\Users\to ja\.spyder-py3\eurotop_data\nn_overtoppingE_q.csv", decimal=".", sep='|')
nn_overtoppingF_q = pd.read_csv(r"C:\Users\to ja\.spyder-py3\eurotop_data\nn_overtoppingF_q.csv", decimal=".", sep='|')
nn_overtoppingG_q = pd.read_csv(r"C:\Users\to ja\.spyder-py3\eurotop_data\nn_overtoppingG_q.csv", decimal=".", sep='|')

