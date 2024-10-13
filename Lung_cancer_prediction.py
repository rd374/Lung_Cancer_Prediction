import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report , accuracy_score
from sklearn.model_selection import train_test_split

import warnings

# data pre- processing 
df=pd.read_csv("C:/Users/DELL/OneDrive/Desktop/SQL/survey lung cancer.csv")
df.head()
# Gender -( Categorical Column) can be converted (1,0)
# age - numerical column(statistical)
## yellow fingers, smoking,chronic disease allergy etc : dummy column \
    # where 1 means symtom  non existent
    # 2 means the symtopm exist
# problem statement : The objective of this project is to develop a machine learning model to 
# predict the likelihood of lung cancer based on a set of given features. Lung 
# cancer is one of the leading causes of cancer-related deaths globally, and 
# early detection is crucial for effective treatment. By leveraging machine 
# learning techniques, we aim to build a predictive model that can assist in 
# the early diagnosis of lung cancer, potentially saving lives and improving 
# patient outcomes

df.head()
df.tail()
# no null values
df.isnull().sum()
# 2 column is object
df.dtypes
# 309 rows and 16 columns 
df.shape
# duplicates

df.drop_duplicates(inplace=True)
df.columns
df.duplicated().sum()

# no missing values no duplicates
# GIVING NUMERICAL VALUES
df['GENDER'] = df['GENDER'].replace(['M', 'F'], [0, 1])
df['LUNG_CANCER']=df['LUNG_CANCER'].replace(['YES','NO'],[1,0])

df.head()
# VALUE COUNTS FOR EACH COLUMN 
Column = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
          'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 
          'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 
          'COUGHING', 'SHORTNESS OF BREATH', 
          'SWALLOWING DIFFICULTY', 'CHEST PAIN', 'LUNG_CANCER','AGE','GENDER']

for col in Column:
    d_type = df[col].dtype
    d_counts = df[col].value_counts()
    print(f"Counts for {col}:")
    print(d_counts)
    print(f"Data type of {col}: {d_type}\n")
    


# Define the list of numeric columns
num_columns = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
               'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 
               'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 
               'COUGHING', 'SHORTNESS OF BREATH', 
               'SWALLOWING DIFFICULTY', 'CHEST PAIN', 'AGE', 'GENDER']

# Set up the plot size for better visibility
plt.figure(figsize=(20, 15))

# Loop through each numeric column and plot
for i, col in enumerate(num_columns):
    plt.subplot(4, 4, i+1)  # Create a grid of 4x4 subplots
    sns.histplot(df[col], bins=10, kde=True)  # Use histplot with KDE (Kernel Density Estimate) for distribution
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
# OUTLIERS
plt.figure(figsize=(20, 15))

for i, col in enumerate(num_columns):
    plt.subplot(4, 4, i+1)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)

plt.tight_layout()
plt.show()




binary_variables = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
                    'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 
                    'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 
                    'COUGHING', 'SHORTNESS OF BREATH', 
                    'SWALLOWING DIFFICULTY', 'CHEST PAIN', 'GENDER']

# Plot count plots for binary variables with Lung Cancer as hue
plt.figure(figsize=(18, 16))
for idx, var in enumerate(binary_variables):
    plt.subplot(4, 4, idx + 1)
    sns.countplot(x=var, hue='LUNG_CANCER', data=df, palette='Set2')
    plt.title(f'Count Plot of {var} with Lung Cancer')
    plt.legend(title='Lung Cancer')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x='LUNG_CANCER', y='AGE', data=df, palette='Set3')
plt.title('Boxplot of AGE with Lung Cancer')
plt.show()
 
plt.figure(figsize = (15,15))
sns.heatmap(df.corr(),annot=True, cbar=True, cmap='Blues', fmt='.1f')

#INFERENCES OF PLOTS:
df['AGE'].describe()

# All the other factors follows a binary distribution (except: age)
# we can see that the age has outliers to and the major chunk of age is in IQR(57-69)
# age follows aleft skewed  distribution
df['AGE'].skew()  #-0.455
# major cancer patients lie in age of 40-60
# flat boxplots because the value is either (1 or 2)
# AGE: There is a weak positive correlation (+0.1) with LUNG_CANCER, suggesting that older age slightly increases the likelihood of having lung cancer.

# SMOKING: It has a low positive correlation (+0.2) with LUNG_CANCER, meaning smoking may have a minor association with lung cancer occurrence. However, it's not strongly indicative on its own.

# YELLOW_FINGERS: This variable shows a positive correlation (+0.3) with LUNG_CANCER, indicating that individuals with yellow fingers (often associated with heavy smoking) may have a higher chance of lung cancer.

# ANXIETY: Anxiety is weakly correlated (-0.1) with LUNG_CANCER, meaning it may not have a significant role in lung cancer prediction.

# PEER_PRESSURE: There is a weak positive correlation (+0.2) between PEER_PRESSURE and LUNG_CANCER.

# CHRONIC DISEASE: This feature has a weak positive correlation (+0.2) with lung cancer, indicating that having a chronic disease may have a slight association with lung cancer.

# ALCOHOL CONSUMING: Shows a weak positive correlation (+0.2) with LUNG_CANCER, but similar to smoking, the association is not strong.

# WHEEZING: Has a moderate correlation (+0.3) with LUNG_CANCER, which may suggest that wheezing could be an indicator or symptom of lung cancer.

# ALLERGY: Shows a slight positive correlation (+0.3) with LUNG_CANCER, though its relationship may be less direct compared to other variables.

# SHORTNESS OF BREATH, COUGHING, and SWALLOWING DIFFICULTY: These respiratory symptoms all have moderate positive correlations with lung cancer (+0.3), indicating that respiratory issues are somewhat linked to the disease.

# CHEST PAIN: Exhibits a slight positive correlation (+0.2) with LUNG_CANCER, implying that chest pain could be related but is not a strong predictor on its own.

# GENDER: It shows a negative correlation (-0.1) with LUNG_CANCER, suggesting a weak inverse relationship, which might indicate that lung cancer prevalence differs slightly between genders in this dataset.

# General Observations:
# No variables have a strong correlation with LUNG_CANCER, as most correlations range between +0.2 and +0.3.
# Variables like YELLOW_FINGERS, WHEEZING, and ALCOHOL CONSUMING seem to have slightly higher positive correlations, making them more relevant features for prediction.
# AGE has a positive but weak correlation, suggesting that age could still be a contributing factor, albeit not strongly.

df.head()
df.columns
# label encoding for logistic regression 
from sklearn.preprocessing import LabelEncoder

# assume 'df' is your pandas DataFrame and 'column_name' is the categorical column you want to encode

le = LabelEncoder()
df['GENDER'] = le.fit_transform(df['GENDER'])
df['YELLOW_FINGERS'] = le.fit_transform(df['YELLOW_FINGERS'])
df['ANXIETY'] = le.fit_transform(df['ANXIETY'])
df['PEER_PRESSURE'] = le.fit_transform(df['PEER_PRESSURE'])
df['CHRONIC DISEASE'] = le.fit_transform(df['CHRONIC DISEASE'])
df['FATIGUE'] = le.fit_transform(df['FATIGUE'])
df['ALLERGY'] = le.fit_transform(df['ALLERGY'])
df['WHEEZING'] = le.fit_transform(df['WHEEZING'])
df['ALCOHOL CONSUMING'] = le.fit_transform(df['ALCOHOL CONSUMING'])
df['COUGHING'] = le.fit_transform(df['COUGHING'])
df['SHORTNESS OF BREATH'] = le.fit_transform(df['SHORTNESS OF BREATH'])
df['SWALLOWING DIFFICULTY'] = le.fit_transform(df['SWALLOWING DIFFICULTY'])
df['CHEST PAIN'] = le.fit_transform(df['CHEST PAIN'])
df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])
df['SMOKING'] = le.fit_transform(df['SMOKING'])

df.head()
pd.set_option('display.max_columns', None)
# transformed the variables into 0 or 1 for logistic regression
#Splitting independent and dependent variables
#From the visualizations, it is clear that in the given dataset, the features GENDER, AGE, SMOKING 
#and SHORTNESS OF BREATH don't have that much relationship with LUNG CANCER. So let's drop those features to make this dataset more clean.

df_new=df.drop(columns=['GENDER','AGE', 'SMOKING', 'SHORTNESS OF BREATH'])
df_new
#Correlation 
cn=df_new.corr()
cmap=sns.diverging_palette(260,-10,s=50, l=75, n=6,
as_cmap=True)
plt.subplots(figsize=(18,18))
sns.heatmap(cn,cmap=cmap,annot=True, square=True)
plt.show()

kot = cn[cn>=.40]
plt.figure(figsize=(12,8))
sns.heatmap(kot, cmap="Blues")

X = df_new.drop('LUNG_CANCER', axis = 1)
y = df_new['LUNG_CANCER']
#Splitting data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.25, random_state=0)
#Fitting training data to the model
from sklearn.linear_model import LogisticRegression
lr_model=LogisticRegression(random_state=0)
lr_model.fit(X_train, y_train)

y_lr_pred= lr_model.predict(X_test)
y_lr_pred

from sklearn.metrics import classification_report, accuracy_score, f1_score
lr_cr=classification_report(y_test, y_lr_pred)
print(lr_cr)
#This model is almost 97% accurate.









