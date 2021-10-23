# =============================================================================
#                             Data Processing
# =============================================================================


# =============================================================================
# packages
# =============================================================================
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# =============================================================================
# Data
# =============================================================================
application_data = pd.read_csv("application_data.csv")
# l , application_data = train_test_split(application_data, test_size=0.2, train_size=0.8, random_state=None)
previous_application = pd.read_csv("previous_application.csv")
#application_data = pd.read_csv(r"C:\Users\shaked\Desktop\לימודים\ML2\project\פרויקט\all_application_data.csv")
#previous_application_new = pd.read_csv(r"C:\Users\shaked\Desktop\לימודים\ML2\project\פרויקט\previous_application_NEW.csv")
# ========================
# Data understanding
# ========================

#getting an idea about the shape of the dataframe
application_data.shape
application_data.info()
application_data.dtypes

# =====================
# EDA
# =====================


plt.figure(figsize=(14,5))

plt.subplot(1,2,1)    
ax = sns.countplot(x = 'TARGET',data=application_data)
plt.title('TARGET')
ax.set(xlabel='TARGET')



sns.boxplot(y=application_data['DAYS_BIRTH'])
plt.show()
application_data['DAYS_BIRTH'].describe()

sns.boxplot(y=application_data['CODE_GENDER'])
plt.show()
application_data['CODE_GENDER'].describe()

sns.boxplot(y=application_data['AMT_REQ_CREDIT_BUREAU_DAY'])
plt.show()

sns.boxplot(y=application_data['CNT_CHILDREN'])
plt.show()
application_data['CNT_CHILDREN'].describe()

sns.boxplot(y=application_data['CNT_FAM_MEMBERS'])
plt.show()
application_data['CNT_FAM_MEMBERS'].describe()


sns.boxplot(y=application_data['AMT_ANNUITY'])
plt.show()
application_data['AMT_ANNUITY'].describe()

# coverting all the 'FLAG' columns from float to bool
col_to_convert_from_numeric_to_bool = [col for col in application_data.columns.to_list() 
                                       if col.startswith('FLAG') == True ] 
for col in col_to_convert_from_numeric_to_bool:
    application_data[col] = application_data[col].astype('bool')


# ==================
# Imbalanced Data
# ==================

# Calculating Imbalance percentage
target0 = application_data.loc[application_data["TARGET"]==0]
target1 = application_data.loc[application_data["TARGET"]==1]
print(f"{round(len(target0)/len(application_data),2)}% didnt late with their payments {round(len(target1)/len(application_data),2)} did late")  

# convert data in days from negative values to positive values
# DAYS Columns

# Divide 'DAYS_BIRTH' by 365 for taking Age
application_data['AGE'] = abs(application_data['DAYS_BIRTH']//365)
# Drop 'DAYS_BIRTH' column
application_data = application_data.drop(['DAYS_BIRTH'],axis=1)

# Divide 'DAYS_EMPLOYED' by 365 for YEAR_EMPLOYED
application_data['YEARS_EMPLOYED'] = abs(application_data['DAYS_EMPLOYED']//365)
# Drop 'DAYS_EMPLOYED' column
application_data = application_data.drop(['DAYS_EMPLOYED'],axis=1)


# =============================================================================
# CRISP-DM: Data Preparation
# =============================================================================
# =============
# Select data
# =============


#plot missing values

NA_col = application_data.isnull().sum().sort_values(ascending = False)
NA_col = NA_col[NA_col.values >(0.30*len(application_data))]
plt.figure(figsize=(20,4))
NA_col.plot(kind='bar', color="#4CB391",ylabel = 'number of missing val')

plt.title('List of Columns & NA counts where NA values are more than 30%')
plt.show()

#getting the percentage of null values in each column
application_data.isnull().sum()/len(application_data)*100

#findidng  removing coloums having greater than 30% null value

emptycol = application_data.isnull().sum()/len(application_data)*100
emptycol = emptycol[emptycol.values>30.0]
emptycol = list(emptycol[emptycol.values>=30.0].index)
application_data.drop(labels=emptycol,axis=1,inplace=True)



cols_irrelevant = ['DAYS_REGISTRATION','FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE',
                   'FLAG_PHONE','FLAG_EMAIL','WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START','LIVE_REGION_NOT_WORK_REGION',
                   'REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY','DAYS_LAST_PHONE_CHANGE',
                  'OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE',
                  'NAME_TYPE_SUITE','CNT_CHILDREN']


# Delete the that irrelevant
application_data = application_data.drop(cols_irrelevant,axis=1)
application_data.shape

# findind all boolean col, check imbalanced , del pure imbalanced

index_of_bool_col = application_data.loc[:, application_data.dtypes == bool] 

# for col in index_of_bool_col:
#     colname = application_data.columns[col]
#     if len(application_data[colname].unique()) == 1 :
#         application_data = application_data.drop(colname,axis=1)
#         print(f"Delete column {col} - only one val")
        

# =============
# Data Cleaning
# =============
' we decide to remove all sample that have more than 4 missing values'
num_of_rows_remove = 0
for rowIndex, row in application_data.iterrows(): #iterate over rows
    c = pd.isnull(row)
    x = list(c)
    s = x.count(True)
    if s > 4 :
        application_data = application_data.drop(rowIndex)
        num_of_rows_remove = num_of_rows_remove + 1
print(f" number of rows that removed is : {num_of_rows_remove}")

application_data.to_csv('application_data_new.csv')
# ============
# outlier
# ============
columns_of_outliers=['AMT_REQ_CREDIT_BUREAU_DAY','AMT_INCOME_TOTAL','DAYS_BIRTH'
                     'DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','CNT_CHILDREN','AMT_ANNUITY']

for col in columns_of_outliers:
    try:
        percentiles = application_data[col].quantile([0.01,0.99]).values
        application_data[col][application_data[col] <= percentiles[0]] = percentiles[0]
        application_data[col][application_data[col] >= percentiles[1]] = percentiles[1]
    except:
        continue
    
    
# ============
# missing values
# ============

# Filling missing values with median according
# to the column and the class he came from

application_data = application_data.reset_index()
numericcol = application_data.select_dtypes(include=np.number).columns.tolist() #numeric col
newdf = application_data.select_dtypes(include=np.number)
newdf_class0 = newdf.loc[newdf['TARGET'] == 0]
newdf_class1 = newdf.loc[newdf['TARGET'] == 1]
a = application_data.copy(deep=True)
a.drop(labels = numericcol,axis=1,inplace=True)
non_numeric = a

for line in range(len(application_data['TARGET'])):
    for col in application_data.columns.to_list() :
        cell = application_data[col][line]
        if cell in newdf_class0.columns.to_list():
            missingValuesFill=newdf_class0[col].mean()
            application_data[col][line] = missingValuesFill
        if cell in newdf_class1.columns.to_list():
            missingValuesFill=newdf_class1[col].mean()
            application_data[col][line] = missingValuesFill
        if cell in non_numeric.columns.to_list():
            missingValuesFill = 'unknow'
            application_data[col][line] = missingValuesFill


# check if all columns dont have nill values
application_data.isnull().sum()/len(application_data)*100
application_data = application_data.dropna()
application_data.to_csv('all_application_data.csv')


# =============================================================================
# DISCRETIZATION OF
# CONTINUOUS VARIABLES
# =============================================================================


# Define function for categorizing AGE_GROUP (Young, Mid age and Senior)
def age_group(x):
    if (x < 40):
        return 'Young'
    elif (x >= 40 and x < 60):
        return 'Mid Age'
    else:
        return 'Senior'


#Creating new column AGE_GROUP
application_data['AGE_GROUP'] = application_data['AGE'].apply(age_group)
application_data = application_data.drop('AGE',axis = 1)

#Creating three income groups - Hign, medium and Low
def credit_group(x):
    if (x < 500000):
        return 'Low'
    elif (x >= 500000 and x < 750000):
        return 'Medium'
    else:
        return 'High'

application_data['CREDIT_GROUP'] = application_data['AMT_CREDIT'].apply(credit_group)
application_data = application_data.drop('AMT_CREDIT',axis = 1)

def income_group(x):
    if (x < 100000):
        return 'Low'
    elif(x >= 100000 and x < 150000):
        return 'Medium'
    else:
        return 'High'

application_data['INCOME_GROUP'] = application_data['AMT_INCOME_TOTAL'].apply(income_group)
application_data = application_data.drop('AMT_INCOME_TOTAL',axis = 1)


application_data['EXT_SOURCE_SCORE'].describe()
def ext_source_group(x):
    if (x < 0.4):
        return 'Low'
    elif (x >= 0.4 and x < 0.6):
        return 'Medium'
    else:
        return 'High'
application_data['EXT_SCORE_CATEGORY'] = application_data['EXT_SOURCE_SCORE'].apply(ext_source_group)
application_data = application_data.drop('EXT_SOURCE_SCORE',axis = 1)
application_data.columns

# =============================================================================
# splitting Data to 2 groups by Target
# =============================================================================


df_current_target_1 = application_data[application_data['TARGET'] == 1] #defaulters
df_current_target_0 = application_data[application_data['TARGET'] == 0] #non-defaulters

# =============================================================================
# Univariate analysis
# =============================================================================

# ===============================
# Unordered categorical variables
# ===============================
#testing if gender implemnt on delaulters 
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)    
ax = sns.countplot(x = 'CODE_GENDER',data=df_current_target_1)
plt.title('Defaulters')
ax.set(xlabel='Gender')

plt.subplot(1,2,2) 
ax = sns.countplot(x = 'CODE_GENDER',data=df_current_target_0)
plt.title('Non-Defaulters')
ax.set(xlabel='Gender')

#testing if Loan type implemnt on delaulters 

plt.subplot(1,2,1)    
ax = sns.countplot(x = 'NAME_CONTRACT_TYPE',data=df_current_target_1)
plt.title('Defaulters')
ax.set(xlabel='Loan type')

plt.subplot(1,2,2) 
ax = sns.countplot(x = 'NAME_CONTRACT_TYPE',data=df_current_target_0)
plt.title('Non-Defaulters')
ax.set(xlabel='Loan type')


#testing if Loan Income  implemnt on delaulters 
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)    
ax = sns.countplot(x = 'NAME_INCOME_TYPE',data=df_current_target_1)
plt.title('Defaulters')
ax.set(xlabel='Income type')
temp = ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment='right')

plt.subplot(1,2,2) 
ax = sns.countplot(x = 'NAME_INCOME_TYPE',data=df_current_target_0)
plt.title('Non-Defaulters')
ax.set(xlabel='Income type')
temp = ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment='right')

#testing if Education type  implemnt on delaulters 

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)    
ax = sns.countplot(x = 'NAME_EDUCATION_TYPE',data=df_current_target_1)
plt.title('Defaulters')
ax.set(xlabel='Education type')
temp = ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment='right')


plt.subplot(1,2,2) 
ax = sns.countplot(x = 'NAME_EDUCATION_TYPE',data=df_current_target_0)
plt.title('Non-Defaulters')
ax.set(xlabel='Education type')
temp = ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment='right')

#testing if Family status  implemnt on delaulters 

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)    
ax = sns.countplot(x = 'NAME_FAMILY_STATUS',data=df_current_target_1)
plt.title('Defaulters')
ax.set(xlabel='Family status')
temp = ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment='right')


plt.subplot(1,2,2) 
ax = sns.countplot(x = 'NAME_FAMILY_STATUS',data=df_current_target_0)
plt.title('Non-Defaulters')
ax.set(xlabel='Family status')
temp = ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment='right')

#we saw that there is no diifrrence between Defaulters and Non-Defaulters so
# we decide to remove this col

application_data = application_data.drop('NAME_FAMILY_STATUS',axis = 1)


# testing if CREDIT_GROUP  amount of the loan  implemnt on delaulters 

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)    
ax = sns.countplot(x = 'CREDIT_GROUP',data=df_current_target_1)
plt.title('Defaulters')

plt.subplot(1,2,2) 
ax = sns.countplot(x = 'CREDIT_GROUP',data=df_current_target_0)
plt.title('Non-Defaulters')


# testing if INCOME_GROUP  amount of the loan  implemnt on delaulters 

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)    
ax = sns.countplot(x = 'INCOME_GROUP',data=df_current_target_1)
plt.title('Defaulters')

plt.subplot(1,2,2) 
ax = sns.countplot(x = 'INCOME_GROUP',data=df_current_target_0)
plt.title('Non-Defaulters')

# ===============================
# Univariate analysis for continious variables
# ===============================

# testing if Loan annuity   implemnt on delaulters 

plt.figure(figsize=(16,6))
plt.subplot(1,2,1) 
plt.title('Defaulters')
sns.distplot(df_current_target_1['AMT_ANNUITY'],hist=False)

plt.subplot(1,2,2) 
plt.title('Non Defaulters')
sns.distplot(df_current_target_0['AMT_ANNUITY'],hist=False)


# ===============================
# Corelation 
# ===============================

# we want to test corelation between our numerical col

numericcol = application_data.select_dtypes(include=np.number).columns.tolist() #numeric col
corr_cols = ['AMT_ANNUITY','AMT_GOODS_PRICE',
             'DAYS_ID_PUBLISH','EXT_SOURCE_2', 'EXT_SOURCE_3','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY',
             'AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR',
             'YEARS_EMPLOYED']

df_corr_target_1 = df_current_target_1[corr_cols]
df_corr_target_1.corr() 
plt.figure(figsize=(10,8))
sns.heatmap(df_corr_target_1.corr(),cmap="YlGnBu",annot=True)

df_corr_target_0 = df_current_target_0[corr_cols]
df_corr_target_0 = df_corr_target_0[corr_cols]
df_corr_target_0.corr() 
plt.figure(figsize=(10,8))
sns.heatmap(df_corr_target_0.corr(),cmap="YlGnBu",annot=True)

df_corr_target = application_data[corr_cols]
df_corr_target.corr() 
plt.figure(figsize=(10,8))
sns.heatmap(df_corr_target.corr(),cmap="YlGnBu",annot=True)

# =============================================================================
# Integrate data
# =============================================================================
import pandasql as ps 
#application_data.drop(application_data.columns[0], axis=1, inplace=True)

previous_application.isnull().sum()/len(previous_application)*100
feture_pre_app = ['SK_ID_CURR','AMT_APPLICATION','AMT_ANNUITY','AMT_CREDIT','AMT_DOWN_PAYMENT',
                  'NAME_CONTRACT_STATUS','NAME_PAYMENT_TYPE','CODE_REJECT_REASON',
                  'DAYS_TERMINATION','RATE_DOWN_PAYMENT',
                  'NAME_CASH_LOAN_PURPOSE',
                  'CODE_REJECT_REASON']
previous_application = previous_application[feture_pre_app]
previous_application = previous_application.dropna()
l , previous_application = train_test_split(previous_application, test_size=0.5, train_size=0.5, random_state=None)

SQL_Query_by_DAYS_TERMINATION = """ SELECT *
                                    FROM  previous_application P
                                    WHERE DAYS_TERMINATION = (SELECT MAX(PP.DAYS_TERMINATION)
                                                              FROM previous_application PP
                                                              WHERE DAYS_TERMINATION < 0
                                                              AND P.SK_ID_CURR = PP.SK_ID_CURR
                                                              GROUP BY SK_ID_CURR)
                                    AND DAYS_TERMINATION < 0
                                    GROUP BY SK_ID_CURR"""
               
previous_application_NEW = ps.sqldf(SQL_Query_by_DAYS_TERMINATION, locals())
previous_application_NEW.to_csv('previous_application_NEW.csv')

SQL_Query_by_DAYS_TERMINATION = """ 
                                SELECT MAX(PP.DAYS_TERMINATION)
                                FROM previous_application PP
                                WHERE DAYS_TERMINATION < 0
                                AND PP.SK_ID_CURR = PP.SK_ID_CURR
                                GROUP BY SK_ID_CURR 
                                """
                                    
X = ps.sqldf(SQL_Query_by_DAYS_TERMINATION, locals())

df = application_data.merge(previous_application_new, left_on='SK_ID_CURR', right_on='SK_ID_CURR')
df.isnull().sum()/len(df)*100
df.drop(df.columns[0], axis=1, inplace=True)
Y = df.TARGET
Y = Y.to_frame()
X = df.drop(['TARGET','Unnamed: 0_y'], axis=1)

# =============================================================================
# Backward Elimination
# =============================================================================

import statsmodels.api as sm

#dummy var
data = pd.get_dummies(X)
data.info()
def backward_elimination(data, target,significance_level = 0.05):
    features = data.columns.tolist()
    while(len(features)>0):
        features_with_constant = sm.add_constant(data[features])
        p_values = sm.OLS(target, features_with_constant).fit().pvalues[1:]
        max_p_value = p_values.max()
        if(max_p_value >= significance_level):
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break 
    return features


features = backward_elimination(data, Y,significance_level = 0.05)
data = data.drop(columns=[col for col in data if col not in features])

data.to_csv('data_after_elimination.csv')
Y.to_csv('Y_after_elimination.csv')





# =============================================================================
#                                Models
# =============================================================================


from pandasql import sqldf
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, ParameterGrid
from sklearn import feature_selection
from sklearn import preprocessing
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
# -- coding: utf-8 --
"""
Created on Sat May  8 10:19:12 2021

@author: Owner
"""

# =============================================================================
# packages
# =============================================================================
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold

# =============================================================================
# Data
# =============================================================================
from sklearn.tree import DecisionTreeClassifier, plot_tree

application_data = pd.read_csv("application_data_new.csv")
#l, application_data = train_test_split(application_data, test_size=0.2, train_size=0.8, random_state=None)

application_data_x = pd.read_csv("data_after_elimination.csv")
application_data_y = pd.read_csv("Y_after_elimination.csv")
#previous_application = pd.read_csv("previous_application.csv")

# =============================================================================
# Modeling
# =============================================================================
#application_data_x = pd.get_dummies(application_data_x[['NAME_CONTRACT_TYPE', 'CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE']],
                            #drop_first=True)

features = application_data_x.keys()
print(features)

# =============================================================================
# Modeling - ANN
# =============================================================================
# First ANN

def First_ANN(datax, datay):
    x_valid = datax
    y_valid = datay
    kf = KFold(n_splits=10)
    kf.get_n_splits(x_valid)
    sum_score_train = 0
    sum_score_test = 0

    for train_index, test_index in kf.split(x_valid):
        X_train, X_test = x_valid.iloc[train_index], x_valid.iloc[test_index]
        Y_train, Y_test = y_valid.iloc[train_index], y_valid.iloc[test_index]

        model = MLPClassifier(random_state=1, max_iter=500, hidden_layer_sizes=(215, 43), alpha=0.0001)
        model.fit(X_train, Y_train)
        train_df = pd.concat([X_train, Y_train], axis=1)
        predictions = train_df.copy()
        predictions['EU_Sales'] = model.predict(X_train)

        print(f"Accuracy: {accuracy_score(y_true=Y_train, y_pred=model.predict(X_train)):2f}")
        sum_score_train += accuracy_score(y_true=Y_train, y_pred=model.predict(X_train))
        print(f"Accuracy: {accuracy_score(y_true=Y_test, y_pred=model.predict(X_test)):2f}")
        sum_score_test += accuracy_score(y_true=Y_test, y_pred=model.predict(X_test))
        plt.plot(model.loss_curve_)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()

    print("Average train Score is: ", round(sum_score_train / 10, 2))
    print("Average test Score is: ", round(sum_score_test / 10, 2))

# Hyper parameter tuning
def ANN_Random_Search(X, Y):
    alphas = [10, 0.1, 0.001, 0.0001]

    learning_rate_init = [0.003, 0.002, 0.001, 0.0001]

    param_grid = {'activation': ['identity', 'logistic', 'relu', 'tanh'],
                  'alpha': alphas,
                  'learning_rate_init': learning_rate_init,
                  'learning_rate': ['constant', 'invscaling', 'adaptive'],
                  'max_iter': [1000]}
    global random_search
    random_search = RandomizedSearchCV(MLPClassifier(random_state=1), param_distributions=param_grid, cv=5,
                                       random_state=42, verbose=True, n_iter=1000, return_train_score=True, refit=True)
    random_search.fit(X, Y)
    best_model = random_search.best_estimator_
    best_parameters = random_search.best_params_
    print(best_parameters)
    plot_search_results_ann(random_search)

def plot_search_results_ann(grid):
    ## Results from random search
    results = grid.cv_results_
    print(results)
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    means_train = results['mean_train_score']
    stds_train = results['std_train_score']

    ## Getting indexes of values per hyper-parameter
    masks = []
    masks_names = list(grid.best_params_.keys())

    # masks_names = ['activation','alpha','learning_rate_init','learning_rate','max_iter']
    print("masks_names ->", masks_names)
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results['param_' + p_k].data == p_v))
        # print(masks)

    params = grid.param_distributions

    print("params -> ", params)

    ## Ploting results
    fig, ax = plt.subplots(1, len(params), sharex='none', sharey='all', figsize=(20, 5))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    for i, p in enumerate(masks_names):
        # for i, p in masks_names:
        print(i)
        print(p)
        m = np.stack(masks[:i] + masks[i + 1:])

        print(m)
        # pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        print("best_parms_mask->", best_parms_mask)
        best_index = np.where(best_parms_mask)[0]
        print(best_index)
        x = np.array(params[p])

        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        y_2 = np.array(means_train[best_index])
        e_2 = np.array(stds_train[best_index])

        print("iter ", i)
        print("X -> ", x)
        print("y_2 ->", y_2)
        print("e_2 ->", e_2)

        xi = np.nan_to_num(x)
        print("this is xi ", xi)

        ei1 = np.nan_to_num(e_1)
        ei2 = np.nan_to_num(e_2)
        yi1 = np.nan_to_num(y_1)
        yi2 = np.nan_to_num(y_2)
        ax[i].errorbar(x=xi, y=yi1, yerr=ei1, linestyle='--', marker='^', label='test')
        ax[i].errorbar(x=xi, y=yi2, yerr=ei2, linestyle='-', marker='^', label='train')
        ax[i].set_xlabel(p.upper())
    plt.legend()
    plt.show()

def Best_ANN(datax, datay):
    x_valid = datax
    y_valid = datay
    kf = KFold(n_splits=10)
    kf.get_n_splits(x_valid)
    sum_score_train = 0
    sum_score_test = 0
    global probs
    probs = pd.DataFrame()
    for train_index, test_index in kf.split(x_valid):
        X_train, X_test = x_valid.iloc[train_index], x_valid.iloc[test_index]
        Y_train, Y_test = y_valid.iloc[train_index], y_valid.iloc[test_index]

        model = MLPClassifier(random_state=1, max_iter=1000, hidden_layer_sizes=(215, 43), alpha=10,
                              activation='tanh', learning_rate='constant', learning_rate_init=0.0001)
        model.fit(X_train, Y_train)
        train_df = pd.concat([X_train, Y_train], axis=1)
        predictions = train_df.copy()
        predictions['TARGET'] = model.predict(X_train)

        print(f"Accuracy: {accuracy_score(y_true=Y_train, y_pred=model.predict(X_train)):2f}")
        sum_score_train += accuracy_score(y_true=Y_train, y_pred=model.predict(X_train))
        print(f"Accuracy: {accuracy_score(y_true=Y_test, y_pred=model.predict(X_test)):2f}")
        sum_score_test += accuracy_score(y_true=Y_test, y_pred=model.predict(X_test))
        # plt.plot(model.loss_curve_)
        # plt.xlabel("Iteration")
        # plt.ylabel("Loss")
        # plt.show()
        prob = pd.DataFrame(np.array(test_index))
        temp = pd.DataFrame(model.predict_proba(x_valid))
        prob['pro 0'] = temp[0]
        prob['pro 1'] = temp[1]
        probs = probs.append(prob, ignore_index=True)


    print("Average train Score is: ", round(sum_score_train / 10, 2))
    print("Average test Score is: ", round(sum_score_test / 10, 2))

    # %%
    probs['dist0'] = abs(probs['pro 0'] - 0.5)
    probs['dist1'] = abs(probs['pro 1'] - 0.5)
    print(probs)

    probs.sort_values('dist0')


# =============================================================================
# Modeling - Random Forest
# =============================================================================

def Forest_FirstAnalysis(datax, datay,Feature_Names):
    x_valid = datax
    y_valid = datay
    kf = KFold(n_splits=10)
    kf.get_n_splits(x_valid)
    print(kf)
    sum_score_train = 0
    sum_score_test = 0
    for train_index, test_index in kf.split(x_valid):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = x_valid.iloc[train_index], x_valid.iloc[test_index]
        Y_train, Y_test = y_valid.iloc[train_index], y_valid.iloc[test_index]

        model = RandomForestClassifier(max_depth=10, random_state=40)
        print("model is: ", model)
        model.fit(X_train, Y_train)
        #plt.figure(figsize=(12,7))
        #plot_tree(model, filled=True, class_names=True, max_depth=5,fontsize=8, label='all', node_ids=True, feature_names=Feature_Names)
        #plt.show()


        train_df = pd.concat([X_train, Y_train], axis=1)
        predictions = train_df.copy()
        predictions['TARGET'] = model.predict(X_train)

        print(f"Accuracy: {accuracy_score(y_true=Y_train, y_pred=model.predict(X_train)):2f}")
        sum_score_train += accuracy_score(y_true=Y_train, y_pred=model.predict(X_train))
        print(f"Accuracy: {accuracy_score(y_true=Y_test, y_pred=model.predict(X_test)):2f}")
        sum_score_test += accuracy_score(y_true=Y_test, y_pred=model.predict(X_test))

    print("Average train Score is: ", sum_score_train / 10)
    print("Average test Score is: ", sum_score_test / 10)

def Grid_Search_Forest(X, Y,Feature_Names):
    list_importence = []
    # number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=500, num=5)]
    # min samples in every leaf
    min_samples_leaf = [int(x) for x in np.linspace(1, 20, num=5)]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    param_grid ={'n_estimators': n_estimators,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    comb = 1
    for list_ in param_grid.values():
        comb *= len(list_)
    print(comb)
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                               param_grid=param_grid, refit=True, return_train_score=True, cv=10)
    grid_search.fit(X,Y)
    best_model = grid_search.best_estimator_
    best_parameters = grid_search.best_params_
    print(best_parameters)

    plot_search_results(grid_search)
    list_importence = best_model.feature_importances_
    print(list_importence)

    ##feature importences
    (pd.DataFrame(list_importence, index=Feature_Names).plot(kind='barh'))
    plt.show()
    plot_search_results(grid_search)
    return best_parameters

# parameters plots
def plot_search_results(grid):

    ## Results from grid search
    results = grid.cv_results_
    print(results)
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    means_train = results['mean_train_score']
    stds_train = results['std_train_score']

    ## Getting indexes of values per hyper-parameter
    masks = []
    masks_names = list(grid.best_params_.keys())

    #masks_names = ['max_depth','criterion','max_features','splitter','min_sample_leaf']
    print("masks_names ->",masks_names)
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results['param_'+p_k].data==p_v))
        #print(masks)

    params = grid.param_grid

    print("params -> ", params)

    ## Ploting results
    fig, ax = plt.subplots(1, len(params), sharex='none', sharey='all', figsize=(20, 5))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    for i, p in enumerate(masks_names):
    #for i, p in masks_names:
        print (i)
        print(p)
        m = np.stack(masks[:i] + masks[i+1:])

        print(m)
        #pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        print("best_parms_mask->",best_parms_mask)
        best_index = np.where(best_parms_mask)[0]
        print(best_index)
        x = np.array(params[p])

        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        y_2 = np.array(means_train[best_index])
        e_2 = np.array(stds_train[best_index])

        print("iter ", i)
        print("X -> ", x)
        print("y_2 ->", y_2)
        print("e_2 ->", e_2)

        xi=np.nan_to_num(x)
        print("this is xi ", xi)

        ei1=np.nan_to_num(e_1)
        ei2=np.nan_to_num(e_2)
        yi1=np.nan_to_num(y_1)
        yi2=np.nan_to_num(y_2)
        ax[i].errorbar(x=xi, y=yi1, yerr=ei1, linestyle='--', marker='^', label='test')
        ax[i].errorbar(x=xi, y=yi2, yerr=ei2, linestyle='-', marker='^', label='train')
        ax[i].set_xlabel(p.upper())
    plt.legend()
    plt.show()

# Best parameters
def bestForest(X,Y,bestParam,Feature_Names):
    listBestP = list(bestParam.values())
    bestBoost = listBestP[0]
    best_leaf=listBestP[1]
    best_estimator=listBestP[2]
    x_valid = X
    y_valid = Y
    kf = KFold(n_splits=10)
    kf.get_n_splits(x_valid)
    print(kf)
    sum_score_train = 0
    sum_score_test = 0
    for train_index, test_index in kf.split(x_valid):
        X_train, X_test = x_valid.iloc[train_index], x_valid.iloc[test_index]
        Y_train, Y_test = y_valid.iloc[train_index], y_valid.iloc[test_index]

        model = RandomForestClassifier(random_state=40,bootstrap=bestBoost,min_samples_leaf= best_leaf,
                                       n_estimators=best_estimator, max_depth=20)

        print("model is: ", model)
        model.fit(X_train, Y_train)
        #plt.figure(figsize=(13, 7))
        #model.estimators_[0].plot_tree(model, filled=True, class_names=True, max_depth=4, fontsize=5, label='all', node_ids=True,
        #          feature_names=Feature_Names)
        #plt.show()

        train_df = pd.concat([X_train, Y_train], axis=1)
        predictions = train_df.copy()
        predictions['TARGET'] = model.predict(X_train)

        print(f"Accuracy: {accuracy_score(y_true=Y_train, y_pred=model.predict(X_train)):2f}")
        sum_score_train += accuracy_score(y_true=Y_train, y_pred=model.predict(X_train))
        print(f"Accuracy: {accuracy_score(y_true=Y_test, y_pred=model.predict(X_test)):2f}")
        sum_score_test += accuracy_score(y_true=Y_test, y_pred=model.predict(X_test))

    print("Average train Score after tuning is: ", sum_score_train / 10)
    print("Average test Score after tuning is: ", sum_score_test / 10)


# =============================================================================
# Modeling - SVM
# =============================================================================
def First_Svm(datax, datay):
    x_valid = datax
    y_valid = datay
    kf = KFold(n_splits=10)
    kf.get_n_splits(x_valid)
    sum_score_train = 0
    sum_score_test = 0

    for train_index, test_index in kf.split(x_valid):
        X_train, X_test = x_valid.iloc[train_index], x_valid.iloc[test_index]
        Y_train, Y_test = y_valid.iloc[train_index], y_valid.iloc[test_index]

        model = SVC(kernel='linear', C=1.0, random_state=42)
        model.fit(X_train, Y_train)
        train_df = pd.concat([X_train, Y_train], axis=1)
        predictions = train_df.copy()
        predictions['TARGET'] = model.predict(X_train)

        print(f"Accuracy: {accuracy_score(y_true=Y_train, y_pred=model.predict(X_train)):2f}")
        sum_score_train += accuracy_score(y_true=Y_train, y_pred=model.predict(X_train))
        print(f"Accuracy: {accuracy_score(y_true=Y_test, y_pred=model.predict(X_test)):2f}")
        sum_score_test += accuracy_score(y_true=Y_test, y_pred=model.predict(X_test))
        print(confusion_matrix(y_true=Y_train, y_pred=model.predict(X_train)))
        print(confusion_matrix(y_true=Y_test, y_pred=model.predict(X_test)))

    print("Average train Score is: ", round(sum_score_train / 10, 2))
    print("Average test Score is: ", round(sum_score_test / 10, 2))

def Grid_SearchSVM(X, Y):
    SVC()
    list_importence = []

    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'gamma': ['scale', 'auto'],
                  'kernel': ['rbf', 'poly', 'sigmoid'],
                  'degree': [2, 3, 4]}

    comb = 1
    for list_ in param_grid.values():
        comb *= len(list_)
    print(comb)
    grid_search = GridSearchCV(estimator=SVC(random_state=42),
                               param_grid=param_grid, refit=True, return_train_score=True, cv=10)
    grid_search.fit(X, Y)
    best_model = grid_search.best_estimator_
    best_parameters = grid_search.best_params_
    print(best_parameters)
    plot_search_results_svm(grid_search)

def Best_Svm(datax, datay):
    x_valid = datax
    y_valid = datay
    kf = KFold(n_splits=10)
    kf.get_n_splits(x_valid)
    sum_score_train = 0
    sum_score_test = 0

    for train_index, test_index in kf.split(x_valid):
        X_train, X_test = x_valid.iloc[train_index], x_valid.iloc[test_index]
        Y_train, Y_test = y_valid.iloc[train_index], y_valid.iloc[test_index]

        model = SVC(kernel='poly', C=100, random_state=42, gamma='scale', degree=2)
        model.fit(X_train, Y_train)
        train_df = pd.concat([X_train, Y_train], axis=1)
        predictions = train_df.copy()
        predictions['TARGET'] = model.predict(X_train)

        print(f"Accuracy: {accuracy_score(y_true=Y_train, y_pred=model.predict(X_train)):2f}")
        sum_score_train += accuracy_score(y_true=Y_train, y_pred=model.predict(X_train))
        print(f"Accuracy: {accuracy_score(y_true=Y_test, y_pred=model.predict(X_test)):2f}")
        sum_score_test += accuracy_score(y_true=Y_test, y_pred=model.predict(X_test))
        print(confusion_matrix(y_true=Y_train, y_pred=model.predict(X_train)))
        print(confusion_matrix(y_true=Y_test, y_pred=model.predict(X_test)))
        print(model.intercept_)

    print("Average train Score is: ", round(sum_score_train / 10, 2))
    print("Average test Score is: ", round(sum_score_test / 10, 2))

    print(model.coef_)
    print(model.intercept_)

def plot_search_results_svm(grid):
    ## Results from random search
    results = grid.cv_results_
    print(results)
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    means_train = results['mean_train_score']
    stds_train = results['std_train_score']

    ## Getting indexes of values per hyper-parameter
    masks = []
    masks_names = list(grid.best_params_.keys())

    # masks_names = ['max_depth','criterion','max_features','splitter','min_sample_leaf']
    print("masks_names ->", masks_names)
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results['param_' + p_k].data == p_v))
        # print(masks)

    params = grid.param_grid

    print("params -> ", params)

    ## Ploting results
    fig, ax = plt.subplots(1, len(params), sharex='none', sharey='all', figsize=(20, 5))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    for i, p in enumerate(masks_names):
        # for i, p in masks_names:
        print(i)
        print(p)
        m = np.stack(masks[:i] + masks[i + 1:])

        print(m)
        # pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        print("best_parms_mask->", best_parms_mask)
        best_index = np.where(best_parms_mask)[0]
        print(best_index)
        x = np.array(params[p])

        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        y_2 = np.array(means_train[best_index])
        e_2 = np.array(stds_train[best_index])

        print("iter ", i)
        print("X -> ", x)
        print("y_2 ->", y_2)
        print("e_2 ->", e_2)

        xi = np.nan_to_num(x)
        print("this is xi ", xi)

        ei1 = np.nan_to_num(e_1)
        ei2 = np.nan_to_num(e_2)
        yi1 = np.nan_to_num(y_1)
        yi2 = np.nan_to_num(y_2)
        ax[i].errorbar(x=xi, y=yi1, yerr=ei1, linestyle='--', marker='^', label='test')
        ax[i].errorbar(x=xi, y=yi2, yerr=ei2, linestyle='-', marker='^', label='train')
        ax[i].set_xlabel(p.upper())
    plt.legend()
    plt.show()

# =============================================================================
# Run - Models
# =============================================================================

# Random Forest
Forest_FirstAnalysis(application_data_x,application_data_y,features)
Grid_Search_Forest(application_data_x,application_data_y,features)
bestF = {'bootstrap': False, 'min_samples_leaf': 1, 'n_estimators': 100}
bestForest(application_data_x,application_data_y,bestF,features)

#ANN
First_ANN(application_data_x,application_data_y)
ANN_Random_Search(application_data_x,application_data_y)
Best_ANN(application_data_x,application_data_y)

#SVM
First_Svm(application_data_x, application_data_y)
Grid_SearchSVM(application_data_x,application_data_y)

# =============================================================================
# Evaluation
# =============================================================================
## Random Forest ##
#Confusion matrix, Accuracy, sensitivity and specificity

def Evaluation_RF(x,y):
    sum_acc =0
    sum_f1= 0
    sum_roc =0
    x_valid = x
    y_valid = y
    kf = KFold(n_splits=10)
    kf.get_n_splits(x_valid)
    print(kf)
    for train_index, test_index in kf.split(x_valid):
        X_train, X_test = x_valid.iloc[train_index], x_valid.iloc[test_index]
        Y_train, Y_test = y_valid.iloc[train_index], y_valid.iloc[test_index]

        model = RandomForestClassifier(random_state=40,bootstrap=True,min_samples_leaf= 10,
                                       n_estimators=500, max_depth=20)

        print("model is: ", model)
        model.fit(X_train, Y_train)

        #train_df = pd.concat([X_train, Y_train], axis=1)
        #predictions = train_df.copy()
        predictions = model.predict(X_test)

        print(Y_test.value_counts())
        cm1 = confusion_matrix(Y_test, predictions)
        plot_confusion_matrix(model, X_test, Y_test)
        plt.show()
        print('Confusion Matrix : \n', cm1)

        total1 = sum(sum(cm1))
        #####from confusion matrix calculate accuracy
        accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
        print('Accuracy : ', accuracy1)
        sum_acc+=accuracy1

        roc_auc=roc_auc_score(Y_test, predictions)
        print('roc auc score: \n', roc_auc)
        sum_roc+=roc_auc

        f1= f1_score(Y_test, predictions)
        print('F1 score: \n', f1)
        sum_f1+=f1

    print("Average F1 score is: ", round(sum_f1 / 10, 4))
    print("Average roc auc score is: ", round(sum_roc / 10, 4))
    print("Average Accuracy is: ", round(sum_acc / 10, 4))

def Evaluation_ANN(x, y):
    sum_acc =0
    sum_f1= 0
    sum_roc =0
    x_valid = x
    y_valid = y
    kf = KFold(n_splits=10)
    kf.get_n_splits(x_valid)
    print(kf)
    for train_index, test_index in kf.split(x_valid):
        X_train, X_test = x_valid.iloc[train_index], x_valid.iloc[test_index]
        Y_train, Y_test = y_valid.iloc[train_index], y_valid.iloc[test_index]

        model = MLPClassifier(max_iter=1000,learning_rate='constant',learning_rate_init=0.003,alpha=0.0001, activation='relu')

        print("model is: ", model)
        model.fit(X_train, Y_train)

        train_df = pd.concat([X_train, Y_train], axis=1)
        predictions = model.predict(X_test)

        cm1 = confusion_matrix(Y_test, predictions)
        plot_confusion_matrix(model, X_test, Y_test)
        plt.show()
        print('Confusion Matrix : \n', cm1)

        total1 = sum(sum(cm1))
        #####from confusion matrix calculate accuracy
        accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
        print('Accuracy : ', accuracy1)
        sum_acc += accuracy1

        roc_auc=roc_auc_score(Y_test, model.predict(X_test))
        print('roc auc score: \n', roc_auc)
        sum_roc += roc_auc

        f1= f1_score(Y_test, model.predict(X_test))
        print('F1 score: \n', f1)
        sum_f1 += f1

    print("Average F1 score is: ", round(sum_f1 / 10, 4))
    print("Average roc auc score is: ", round(sum_roc / 10, 4))
    print("Average Accuracy is: ", round(sum_acc / 10, 4))


Evaluation_RF(application_data_x,application_data_y)
Evaluation_ANN(application_data_x,application_data_y)

# =============================================================================
# Improve selected model
# =============================================================================
# class count
new_data = pd.concat([application_data_y, application_data_x], axis=1)
class_count_0, class_count_1 = new_data['TARGET'].value_counts()
color=['g','b']
new_data['TARGET'].value_counts().plot(kind='bar',stacked=True, color=color, legend=False, figsize=(12, 4), title='count (target)')
plt.show()

# Separate class
class_0 = new_data[application_data_y['TARGET'] == 0]
class_1 = new_data[application_data_y['TARGET'] == 1]# print the shape of the class
print('class 0:', class_0.shape)
print('class 1:', class_1.shape)

class_0_under = class_0.sample(class_count_1)

test_under = pd.concat([class_0_under, class_1], axis=0)

print("total class of 1 and 0:",test_under['TARGET'].value_counts())# plot the count after under-sampeling
test_under['TARGET'].value_counts().plot(kind='bar', stacked=True, color=color, legend=False, figsize=(12, 4), title='count (target)')
plt.show()
test_under= shuffle(test_under)

features_new =['AMT_CREDIT_x', 'AMT_ANNUITY_x', 'AMT_GOODS_PRICE',
       'REGION_POPULATION_RELATIVE', 'DAYS_ID_PUBLISH', 'REGION_RATING_CLIENT',
       'REGION_RATING_CLIENT_W_CITY', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
       'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_16',
       'FLAG_DOCUMENT_18', 'AMT_REQ_CREDIT_BUREAU_WEEK',
       'AMT_REQ_CREDIT_BUREAU_QRT', 'AGE', 'YEARS_EMPLOYED', 'AMT_APPLICATION',
       'AMT_ANNUITY_y', 'AMT_CREDIT_y', 'AMT_DOWN_PAYMENT', 'DAYS_TERMINATION',
       'RATE_DOWN_PAYMENT', 'CODE_GENDER_F', 'FLAG_OWN_CAR_Y',
       'NAME_INCOME_TYPE_Commercial associate',
       'NAME_INCOME_TYPE_State servant',
       'NAME_EDUCATION_TYPE_Higher education',
       'NAME_EDUCATION_TYPE_Incomplete higher', 'NAME_FAMILY_STATUS_Married',
       'NAME_FAMILY_STATUS_Separated',
       'NAME_FAMILY_STATUS_Single / not married', 'NAME_FAMILY_STATUS_Widow',
       'NAME_HOUSING_TYPE_House / apartment', 'ORGANIZATION_TYPE_Bank',
       'ORGANIZATION_TYPE_Industry: type 9', 'ORGANIZATION_TYPE_Military',
       'ORGANIZATION_TYPE_Trade: type 6',
       'ORGANIZATION_TYPE_Transport: type 3', 'ORGANIZATION_TYPE_XNA']
#ANN
First_ANN(test_under[features_new],test_under['TARGET'])
ANN_Random_Search(test_under[features_new],test_under['TARGET'])
Best_ANN(test_under[features_new],test_under['TARGET'])

#Random Forest

Forest_FirstAnalysis(test_under[features_new],test_under['TARGET'], features_new)
Grid_Search_Forest(test_under[features_new],test_under['TARGET'],features_new)
bestF = {'bootstrap': True, 'min_samples_leaf': 10, 'n_estimators': 500}
bestForest(test_under[features_new],test_under['TARGET'],bestF,features)

# Evaluation after under sampling
Evaluation_RF(test_under[features_new],test_under['TARGET'])
Evaluation_ANN(test_under[features_new],test_under['TARGET'])

#Recondider Features
features_new2 =['AMT_CREDIT_x', 'AMT_ANNUITY_x', 'AMT_GOODS_PRICE',
       'REGION_POPULATION_RELATIVE', 'DAYS_ID_PUBLISH', 'REGION_RATING_CLIENT',
       'REGION_RATING_CLIENT_W_CITY', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
       'FLAG_DOCUMENT_3',  'AMT_REQ_CREDIT_BUREAU_WEEK',
       'AMT_REQ_CREDIT_BUREAU_QRT', 'AGE', 'YEARS_EMPLOYED', 'AMT_APPLICATION',
       'AMT_ANNUITY_y', 'AMT_CREDIT_y', 'AMT_DOWN_PAYMENT', 'DAYS_TERMINATION',
       'RATE_DOWN_PAYMENT', 'CODE_GENDER_F', 'FLAG_OWN_CAR_Y',
       'NAME_INCOME_TYPE_Commercial associate',
       'NAME_INCOME_TYPE_State servant',
       'NAME_EDUCATION_TYPE_Higher education',
        'NAME_FAMILY_STATUS_Married',
       'NAME_FAMILY_STATUS_Separated',
       'NAME_FAMILY_STATUS_Single / not married', 'NAME_FAMILY_STATUS_Widow',
       'NAME_HOUSING_TYPE_House / apartment', 'ORGANIZATION_TYPE_Bank',
       'ORGANIZATION_TYPE_Industry: type 9',
       'ORGANIZATION_TYPE_Trade: type 6', 'ORGANIZATION_TYPE_XNA']

Evaluation_RF(test_under[features_new2],test_under['TARGET'])
Evaluation_ANN(test_under[features_new],test_under['TARGET'])

