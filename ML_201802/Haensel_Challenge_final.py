
# coding: utf-8

# # HAENSEL-AMS Challenge

# ### 1. Importing Libraries

# In[1]:


#importing the basic libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# ### 2. Loading Dataset and Checking the Data

# In[2]:


df = pd.read_csv('sample.csv',header = None, skipinitialspace = True ) # assuming here data is in the working directory.


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.head()


# In[6]:


# are there any constant columns? 11 cols are constant, store indices to drop in data cleaning step.

df_unique = df.loc[:,df.apply(pd.Series.nunique) != 1].columns
print(df_unique.shape)
const_cols = set(df.columns)-set(df_unique)
print(const_cols)


# In[7]:


df.describe()


# In[8]:


# 291 columns are int and the majority appear to be binary. How many columns are binary? 

bin_cols = [col for col in df.drop(295, axis=1) 
             if df.drop(295, axis=1)[col].dropna().value_counts().index.isin([0,1]).all()]
len(bin_cols)


# In[9]:


# 288 columns are binary and indices are stored in bin_cols. Which columns are not binary ?

nonbin_cols = set(df.drop(295, axis=1).columns) - set(bin_cols)
nonbin_cols


# In[10]:


# What are the indices of the columns that are floats?

float_cols = df.loc[:, df.dtypes == float].columns


# In[11]:


#What is the distribution of these non-binary columns/attributes?

df.loc[:,nonbin_cols].describe()


# In[12]:


# does the data have any missing data in the form of nans?

df.dropna().shape


# In[16]:


# What is the representation of each target class in the data?

df[295].value_counts()


# In[17]:


# What is the representation of each target class in the data in percentage?

(df[295].value_counts()/df.shape[0])*100


# In[18]:


# Plotting the distribution of target classes

sns.countplot(x=df[295], data=df, palette='Set1')


# ** From the above data check the following observations can be stated:**
# 
# - The data has no missing values in the form of NaNs.
# - There are eleven constant columns. These will be dropped in the data cleaning step.
# - The vast majority (288/295) of the attributes columns are binary. This can be because the columns represent dummy variables
#   of categorical attributes or because they represent pixels of a black and white image(The binary columns only).
# - There are four real-number/float attributes. Their distribution for every target class should be checked.
# - There are 3 non-binary integer attributes. Their distributions suggest they have extreme outliers. This must
#   be further investigated in the EDA step.
# - The response classes A to E are very unbalanced, with class C appearing in more than 70% of the instances. This is expected
#   to affect the performance of ML models. 

# ### 3. Exploratory Data Analysis and Data Cleaning

# In[19]:


# Taking another look at the distribution of the non-binary columns

df.loc[:,(3,4,23,36,43,64,294)].describe()


# In[11]:


# renaming the non-binary columns for easy reference

df=df.rename(columns = {295:'response', 3:'float1', 43:'float2', 64:'float3', 294:'float4'})


# In[12]:


df=df.rename(columns = {4:'col4', 23:'col23', 36:'col36'})


# In[22]:


df.head()


# In[23]:


# Checking col4's distribution: It is mostly equal to 0 (87% of the time) with outliers between 1 and 10

df.boxplot(column='col4')
len(df[df['col4']==0].col4)


# In[24]:


# Plotting the distribution of col4 per target class. I think it should be dropped since outliers are in all classes.

sns.set_style("whitegrid")
print(df[df['col4'] == 0].shape)
sns.boxplot(x='col4', y='response', hue='response', data=df)


# In[33]:


# Checking col23's distribution: It is almost constant at -1 (99.6% of the time)

df.boxplot(column='col23')
len(df[df['col23']== -1].col23)


# In[35]:


# Plotting the distribution of col23 per target class. I think it should be dropped as well for the same reason as col4.

print(df[df['col23'] == -1].shape)
sns.boxplot(x='col23', y='response', hue='response', data=df)


# In[47]:


# Checking col36's distribution: There is an extreme positive outlier(s) that is messing the scale. 

df.boxplot(column='col36')
print(len(df[df['col36']== 1].col36))
print(len(df[df['col36']== 0].col36))


# In[25]:


# Plotting the distribution of col36 per target class. The column takes binary values 80% of the time. It took a few 
# iterations to see that the remaining 20% is distributed between 0 and 20 and 1.5 % is extreme values. 
# Not knowing what the attribute is and whether it is really binary or not, I will only trim/drop the extreme outliers 
# so as not to negatively impact normalizing/rescaling this column.

print(df[df['col36'] < 20000].shape)
print(df[df['col36'] < 20].shape)
print(df[df['col36'] < 0].shape)
sns.boxplot(x='col36', y='response', hue='response', data=df[(df['col36'] < 20) & (df['col36'] >= 0)]) 


# In[49]:


# Plotting the distributions of the float1 attribute

sns.boxplot(x='float1', y='response', hue='response', data=df)
print(df[df['float1'] < 50000].shape)


# In[129]:


sns.boxplot(x='float2', y='response', hue='response', data=df)


# In[50]:


sns.boxplot(x='float3', y='response', hue='response', data=df)
print(df[df['float3'] < 1500].shape)


# In[51]:


sns.boxplot(x='float4', y='response', hue='response', data=df)
print(df[df['float4'] < 1500].shape)


# In[ ]:


#In all four float columns, there are outliers. However, I think not knowing the nature of the attributes I will keep all the
# data for these columns, especially that there are no outliers orders of magnitude greater than the mean as in col36 for 
# example.


# In[53]:


# Are the float attributes correlated?

float_df=df[['float1','float2','float3','float4','response']]


# In[54]:


float_df.sample(10)


# In[55]:


plt.figure(figsize=(8,6))
sns.pairplot(float_df,hue='response',palette='Set1')


# In[56]:


# Atrributes float1 and float2 are moderately correlated, while float3 and float4 are strongly correlated.

sns.heatmap(float_df.corr())


# In[13]:


# Cleaning the data. Now df contains the cleaned data.

df.drop(const_cols, axis=1, inplace=True)
df.drop(['col4', 'col23'], axis=1, inplace=True)
df=df[(df['col36'] < 20) & (df['col36'] >= 0)]
df.shape


# ### 4. Building ML Models
# 
# ** Performance Evaluation/Validation Metrics: Due to the target classes representation imbalance in the data, the following models are evaluated based on their Confusion Matrix and F1-score. **
# 
# - The **Confusion Matrix** will show how each model is performing in terms of identifying/predicting each class as opposed to accuracy, which can just reflect the underlying imbalanced distribution of the classes. 
# 
# - The **average F1-score** is the weighted average of the Precision and Recall (sensitivity), and hence is more descriptive of the model performance, assuming that the cost of a false positive and a false negative are equal.
# 

# ### 4.1 Random Forest
# 
# ** To get a base model for the data, Random Forest is used because:**
# 
# - RF can work with a mix of binary and float attributes without the need to standardize/scale the attributes.
# - RF can work with the relatively large number of attributes without applying dimensionality reduction first.
# - RF is robust against correlated attributes, which are present in this data, at for the float attributes.

# In[14]:


# Splitting the data to test and training

from sklearn.model_selection import train_test_split

X = df.drop('response', axis=1)
y = df['response']


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[30]:


# Importing RF classifier and creating an instant with number of estimators set to 100. I will not assign weights to classes
# in this step to get a base model

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
#rfc.fit(X_train, y_train)


# In[31]:


# Fitting the model

rfc.fit(X_train, y_train)


# In[24]:


from sklearn.metrics import classification_report,confusion_matrix


# In[33]:


# Predicting off the test set and printing metrics. 

rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))


# In[34]:


# The obtained model has a good Presicion and Recall for Class C, the majority class. However, the performance for the other 
# classes is very bad, which is expected as a result of the class imbalance especially for class E and A.

# I will try again, but with the attribute class_weight set to "balanced" mode, which uses the values of y_train to automatically
# adjust weights inversely proportional to class frequencies in the input data.

# The second model did not provide any enhancement, but was 1% worse in terms if the average F1-score.

rfc = RandomForestClassifier(n_estimators=100, class_weight="balanced")
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))


# ### 4.2 K-Nearest Neighbor
# 
# ** KNN will be applied to the data after standardizing (z-score) all the attributes. The tunable parameter K is set initially to 10. KNN has the advantage of being a simple ML technique with low computational complexity. Hence it can be used to train relatively large and high dimensional datasets that grows progressively over time. However, its downside is that it is expensive in terms of memory since the trained model is represented by the entire dataset.**

# In[15]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

# scaling data so that all attributes have zero mean and unit std. This step is vital for KNN since 5 of the attributes have much
# larger ranges and variances than the binary attributes.
scaled_df = scale(df.drop('response', axis=1))


# In[38]:


X = scaled_df
y = df['response']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[39]:


# Fitting the model

knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train,y_train)


# In[40]:


# Predicting off the test test and printing metrics

pred = knn.predict(X_test)

print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred))


# ### 4.3. Principal Component Analysis + Support Vector Machine Model
# 
# ** The next classifier I will try is Support Vector Machine. Due to the relatively high number of attributes (and also records)
# PCA will be used to reduce the dimensions of the data before applying SVM.**

# In[16]:


# Splitting the data

X = scaled_df
y = df['response']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[17]:


from sklearn.decomposition import PCA

pca = PCA(n_components=281) # we are getting all the principal components first to then see how many to use later before SVM,
pca.fit(X_train)


# In[18]:


# Now checking to see the amount of variance the addition of each PC explains

var= pca.explained_variance_ratio_
var_cs=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
print(var_cs)


# In[19]:


# The rate of increase in the amount of total variation explained with PCs is slow. 
# I will take a number of PCs that explains 80% of the variation

pca = PCA(n_components=.80, svd_solver='full') 
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_train_pca.shape


# In[20]:


# so we reduced the dimensions here by almost 40%. I suspect that 45K records of 174 attributes will be too computationally 
# expensive to run in a reasonable time using SVM but I will try.
# applying PCA to test data

X_test_pca = pca.transform(X_test)
print(X_test_pca.shape)


# In[21]:


# importing SVM and creating an SVM instance, first without class_weight set to "balanced"

from sklearn.svm import SVC

svc_model = SVC()


# In[22]:


# fitting the model

svc_model.fit(X_train_pca,y_train)


# In[25]:


# Predicting off the the test set

predictions = svc_model.predict(X_test_pca)

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))


# In[108]:


# importing SVM and creating an SVM instance with class_weight set to "balanced".

svc_model = SVC(class_weight="balanced")


# In[109]:


# fitting the SVM model

svc_model.fit(X_train_pca,y_train)


# In[110]:


# Predicting off the the test set

predictions = svc_model.predict(X_test_pca)

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))


# ** Compared to the RF Model, the average F1-score is 8% worse. However, the F1-score of the underepresented classes improved 
#    significantly, except for class A which has only 1% representation. The next steps can be:**
#    
#    - Perform a Grid Search for tuning the parameters C and gamma for the higest f1 score. However, this is computationally 
#      very expensive with this number of attributes and records. This would be the code for this:
#      
#      - #param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
#      - #from sklearn.model_selection import GridSearchCV
#      - #grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3, scoring= ['f1_weighted','f1_sampled'])
#      - #grid.fit(X_train_pca,y_train)
#      - #grid.best_params_
#      - #grid.best_estimator_
#      - #grid_predictions = grid.predict(X_test_pca)
#      - #print(confusion_matrix(y_test,grid_predictions))
#      - #print(classification_report(y_test,grid_predictions))
#    
#    
#    - Before splitting the data and applying PCA, a RandomUnderSampler from the imblearn can be applied to the data.
#      Since class A is the least represented in the data with around 800 instances, the data after balancing out with random
#      undersampling will shrink siginifantly to achieve a 1:1:1:1:1 default ratio. PCA is then applied and the GridSearch for 
#      tuning SVM parameters C and gamma can be used. The following is the code lines to be added before fitting PCA:
#     
#     - #from imblearn.under_sampling import RandomUnderSampler
#     - #rus = RandomUnderSampler(random_state=101)
#     - #X_res, y_res = rus.fit_sample(X, y)
#    
#    

# ### 4.3 Artificial Neural Network

# In[25]:


df_copy = df.copy()


# In[26]:


# importing libraries

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


# In[27]:


# encoding target classes as integers and then hot-one encoding

# Encode as integers from 0 to 4
encoder = LabelEncoder()
encoder.fit(df['response'])
encoded_Y = encoder.transform(df['response'])

# convert integers to dummy variables (i.e. one hot encoded)
y_nn = np_utils.to_categorical(encoded_Y)


# In[29]:


#Scaling non-binary columns to the range (0,1). This will not affect the values in the binary columns.

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X=df.drop(['response'], axis=1)
X_nn = scaler.fit_transform(X)


# In[78]:


# Now the all  the attributes columns contain floats between 0 and 1

pd.DataFrame(X_nn).head()


# In[30]:


# Splitting the data

X_train, X_test, y_train, y_test = train_test_split(X_nn, y_nn, test_size=0.30, random_state=101) 


# In[31]:


X_train.shape


# In[32]:


# Defining the base model, which is the input for the KerasClassifier we will create in the next step. 
# The base model defines the topology of the ANN: 
# - 1 hidden layer layer with 80 perceptrons (according to the formula rule of thumb --> Nh= Nt(=45603)/2*(Ni(=282)+No(=5))). 
#   RELU (rectifier) function is used for perceptron activation.
# - Ouput layer with 5 perceptrons, 1 for each target class. A SoftMax function is used to match the one-hot o/p encoding.
# - Adam gradient descent optimization algorithm is used
# - A Cross-Entropy function is used as the error/loss measure.

from keras import metrics
def baseline_model():
    model = Sequential()
    model.add(Dense(80, input_dim=282, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', metrics.categorical_accuracy])
    return model


# In[33]:


# creating an instance of the KerasClassifier. Inputs are:
# - Defined baseline_model
# - Number of iterations/epochs to run. 
# - batch size. For the size of training data we have 200 is chosen

estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=200, verbose=1)


# In[34]:


estimator.fit(X_train, y_train)


# In[35]:


predictions = estimator.predict(X_test)


# In[36]:


predictions_label = encoder.inverse_transform(predictions)
y_test_int = np.argmax(y_test,1)
y_test_label = encoder.inverse_transform(y_test_int)


# In[37]:


print(confusion_matrix(y_test_label,predictions_label))

print(classification_report(y_test_label,predictions_label))


# In[38]:


# The model performed in terms of average f1-score the same as the base model with RF. Next Steps can be:

# - Tune parameters of the ANN further: number of training epochs and number of perceptrons in hidden layer.
# - Add another hidden layer. This may improve performance but will be difficult to train.
# - Address class imbalance by using a random under sampler


# In[39]:


# I will try the thrid option here, which is to under sample and then repeat training and testing.

from imblearn.under_sampling import RandomUnderSampler 

rus = RandomUnderSampler(random_state=101)
X = df.drop('response', axis = 1)
y = df['response']
X_res, y_res = rus.fit_sample(X, y)
dfy_res = pd.DataFrame(y_res)
dfy_res[0].value_counts()


# In[40]:


# Now we have a balanced data set of 4170 records and 282 attributes.

# Repeating prepping the inputs to the ANN and splitting the data
encoder = LabelEncoder()
encoder.fit(y_res)
encoded_Y = encoder.transform(y_res)

# convert integers to dummy variables (i.e. one hot encoded)
y_nn = np_utils.to_categorical(encoded_Y)

#Scaling non-binary columns to the range (0,1). This will not affect the values in the binary columns.

scaler = MinMaxScaler()
X_nn = scaler.fit_transform(X_res)

# Splitting the data

X_train, X_test, y_train, y_test = train_test_split(X_nn, y_nn, test_size=0.30, random_state=101) 

X_train.shape


# In[41]:


# using the same ANN model, we train and test. 
# The resulting performance was an f1-score of 0.28. This is because the ANN overfitted on the training data (acurracy 0.9884)
# and hence did not learn the classes properly.

estimator.fit(X_train, y_train)

predictions = estimator.predict(X_test)

predictions_label = encoder.inverse_transform(predictions)
y_test_int = np.argmax(y_test,1)
y_test_label = encoder.inverse_transform(y_test_int)

print(confusion_matrix(y_test_label,predictions_label))

print(classification_report(y_test_label,predictions_label))

