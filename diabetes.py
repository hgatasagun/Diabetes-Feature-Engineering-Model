###################################################################################
# Building a Diabetes Prediction Model through Data Analysis and Feature Engineering
####################################################################################

# A machine learning model is requested to be developed that can predict whether individuals are diabetic or not when
# their features are provided. Before developing the model, it is expected to perform the necessary data analysis and
# feature engineering steps.

##############################################################
# 1. Business Problem
##############################################################

# The dataset is a part of a large dataset maintained at the National Institute of Diabetes and Digestive and Kidney
# Diseases in the USA. It is used for a diabetes study conducted on Pima Indian women aged 21 and above living in
# Phoenix, the 5th largest city in the state of Arizona. The target variable is defined as "outcome", where 1
# indicates a positive diabetes test result and 0 indicates a negative result.

# Variables:
# Pregnancies: Number of pregnancies
# Glucose: 2-hour plasma glucose concentration measured during an oral glucose tolerance test
# Blood Pressure: Diastolic blood pressure (mm Hg)
# SkinThickness: Triceps skinfold thickness
# Insulin: 2-hour serum insulin (mu U/ml)
# DiabetesPedigreeFunction: Diabetes pedigree function (a function related to the diabetes history of relatives and
# the genetic relationship)
# BMI: Body mass index
# Age: Age in years
# Outcome: Presence (1) or absence (0) of the disease

###############################################################
# 2. Data Preparation
###############################################################

# 2.1. Importing libraries
##############################################
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv('diabetes.csv')
df.columns = [col.upper() for col in df.columns] # converting each column name to uppercase

# 2.2. Data understanding
##############################################
def check_df(dataframe, head=5):
    print('################# Columns ################# ')
    print(dataframe.columns)
    print('################# Types  ################# ')
    print(dataframe.dtypes)
    print('##################  Head ################# ')
    print(dataframe.head(head))
    print('#################  Shape ################# ')
    print(dataframe.shape)
    print('#################  NA ################# ')
    print(dataframe.isnull().sum())
    print('#################  Quantiles ################# ')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99]).T)

check_df(df)


# 2.3. Capturing numeric and categorical variables
##############################################
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Returns three lists: categorical, numeric, and categorical-like but
    cardinal variables from the dataframe.

    Parameters
    ------
        dataframe: dataframe
                    The dataframe from which variable names are to be taken.
        cat_th: int, optional
                    Class threshold value for numeric but categorical variables.
        car_th: int, optinal
                    Class threshold value for categorical but cardinal variables.

    Returns
    ------
        cat_cols: list
                    List of categorical variables.
        num_cols: list
                    List of numeric variables.
        cat_but_car: list
                    List of categorical-like but cardinal variables.

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables.
        num_but_cat is inside cat_cols.

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# 2.4. Analysing numeric and categorical variables
#############################################
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    if df[col].dtypes == 'bool':
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df, col, plot=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot = True)


# 2.5. Analysing target variables, 'OUTCOME'
#############################################
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: 'mean'}), end='\n\n\n')

for col in num_cols:
    target_summary_with_num(df, 'OUTCOME', col)


# 2.6. Detecting outliers and analysis
####################################################################
# Note: In this study, a random forest algorithm will be used and Random forests can handle outliers quite well
# due to the ensemble nature of the algorithm. Therefore, the section related to outlier removal can be skipped,
# and you can proceed to the next section.

# a) Setting outlier threshold value (by using the IQR method):
###############################################################
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# b) Detection of outlier presence (True/False):
################################################
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))


# c) Identifying outlying variables:
####################################
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    outlier_df = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))]

    if outlier_df.shape[0] > 10:
        print(outlier_df.head())
    else:
        print(outlier_df)

    if index:
        outlier_index = outlier_df.index
        return outlier_index

    return outlier_df.shape[0]

for col in num_cols:
    print(col, grab_outliers(df, col))


# d) Removing outliers using the LOF method:
############################################
clf = LocalOutlierFactor(n_neighbors=20) # the number of neighbors for the algorithm
clf.fit_predict(df) # detecting outliers

df_scores = clf.negative_outlier_factor_ # obtaining outlier scores
np.sort(df_scores)[0:5] # the worst 5 outliers

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-') # visualizing outlier scores
plt.show()

th = np.sort(df_scores)[10] #  the outlier threshold value - the observation at index 10
# -1.7062264314830031

rows_to_drop = df[df_scores < th].index
df.drop(axis=0, index=rows_to_drop, inplace=True) # removing outliers
df.shape # the number of remaining observations after removing the outliers


# 2.7. Analysing missing values
################################
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

# Note: While there are no missing observations in the dataset, it was observed that certain columns
# like BMI, insulin, skin thickness, blood pressure have been assigned zero values.
# Given that these columns cannot logically hold zero values, it is presumed that these zeros represent
# missing data. Consequently, these zero values have been replaced with NaN values during the analysis.

zero_valued_columns = df.columns[df.eq(0).any()] # for pregnancies and outcome variables, zero values are possible.

na_val = ['BLOODPRESSURE', 'SKINTHICKNESS', 'INSULIN', 'BMI']
df[na_val] = df[na_val].replace(0, np.nan)

df.isnull().sum()


# 2.8.Correlation analysis
##########################
def high_correlated_cols(dataframe, plot = False, corr_th = 0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap='RdBu')
        plt.show(block=True)
    return drop_list

high_correlated_cols(df, plot = True)