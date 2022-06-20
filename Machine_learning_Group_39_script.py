# -*- coding: utf-8 -*-
"""MLchallenge_final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ll-6Ye4B4yFdaba6cBjirB1d1wVKEQFf

# **MACHINE LEARNING CHALLENGE - GROUP 39 **


NOTE: all of this script was done using Google Colab noteboook IDE. We decided to use this IDE since it would make collaboration easier, and we are able to run the code in a faster machine rented from Google

To further read about Google Colab and how it works, please refer to this page: https://colab.research.google.com/notebooks/intro.ipynb

However, if you want to run this code on Jupyter or other Python IDE, please make sure: 

*   To change the working directory to where you store the train-1.json and test.json 
*   To not load the package from google.colab 

*   To save the predicted.json accordingly

Thank you for reading!
"""

#Use this code to mount your google drive with google colab if needed
from google.colab import drive
drive.mount('/content/drive')

#import all packages needed
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords # Import the stop word list
from tabulate import tabulate #to print dataframe prettier and easier to look at
from sklearn import linear_model
from sklearn.model_selection import KFold #to import k-fold cross validation
from sklearn.model_selection import train_test_split #to split train dataset into train + validation set
from sklearn.ensemble import RandomForestRegressor #to input random forest
from sklearn.metrics import r2_score #to calculate r2_score
from sklearn.preprocessing import StandardScaler #to rescale the variables preventing skewness
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MultiLabelBinarizer #to encode categorical variables such as topics, fields of study

"""**STEP 1**: Processing the Train and the Test set
(Note: please make sure to have train-1.json and test.json in your working directory)
"""

#1) Load the train dataset
import json
t = open("/content/drive/MyDrive/Colab Notebooks/train-1.json")
train = json.load(t)
#convert train dataset into a panda dataframe
df = pd.DataFrame(train)
#find duplicate values 
print(pd.concat(g for _, g in df.groupby('doi') if len(g) > 1)) #credit: https://stackoverflow.com/questions/14657241/how-do-i-get-a-list-of-all-the-duplicate-items-using-pandas-in-python
#remove duplicate value: keep the first occurence 
df.drop_duplicates(subset = 'doi',keep = 'first',inplace = True)
#drop NAs
df.dropna()
#glance at train df
print(tabulate(df.head(), headers='keys', tablefmt='psql')) #credit: https://stackoverflow.com/questions/18528533/pretty-printing-a-pandas-dataframe

#2) Processing the test dataset 
#load the test dataset
t1 = open("/content/drive/MyDrive/Colab Notebooks/test.json")
test = json.load(t1)
#convert test dataset into a panda dataframe
df1 = pd.DataFrame(test)
#remove duplicate value: keep the first occurence 
df1.drop_duplicates(subset = 'doi',keep = 'first',inplace = True)
#drop NAs
df1.dropna()
#glance at test df
print(tabulate(df1.head(), headers='keys', tablefmt='psql')) #credit: https://stackoverflow.com/questions/18528533/pretty-printing-a-pandas-dataframe

"""**STEP 2:** Pre-processing the text data"""

#1) Define the clean text function
def clean_text(text_to_clean):
    text_to_clean = re.sub( '[^a-zA-Z0-9]', ' ', str(text_to_clean)) # subs charact in the brackets
    text_to_clean = re.sub( '\s+', ' ', str(text_to_clean)).strip() ## subs tabs,newlines and "whitespace-like"
    words = text_to_clean.lower().split() ## convert to lowercase split indv words
    stops = set(stopwords.words('english')) #converting stop words to set
    meaningful_words = [w for w in words if not w in stops] # removing stop words
    return(" ".join(meaningful_words))

#credit: https://github.com/berkurka/Reddit-Classifier/blob/master/Notebooks/02%20EDA.ipynb

#2) clean title, abstract, venue of the train df and make lists of each of the variable:
df['title'] = df['title'].apply(lambda x: clean_text(x))
title = df['title'].tolist()
df['abstract'] = df['abstract'].apply(lambda x: clean_text(x))
abstract = df['abstract'].tolist()
df['venue'] = df['venue'].apply(lambda x: clean_text(x))
venue = df['venue'].tolist()

#3) clean title, abstract, venue and make lists of each of the variable:
df1['title'] = df1['title'].apply(lambda x: clean_text(x))
title_test = df1['title'].tolist()
df1['abstract'] = df1['abstract'].apply(lambda x: clean_text(x))
abstract_test = df1['abstract'].tolist()
df1['venue'] = df1['venue'].apply(lambda x: clean_text(x))
venue_test = df1['venue'].tolist()

#4) combine train and test df title + abstract for tf-idf (later step)
title_combined = title + title_test
abstract_combined = abstract + abstract_test

#5) get a list of authors in train df 
authors = df['authors'].tolist()

#6) get a list of authors in test df
authors_test = df1['authors'].tolist()

#7) clean text for topic in train df
topics = df['topics'].tolist()
topics_1 = []
for i in topics:
    item = []
    for y in i:
        item.append(clean_text(y))
    topics_1.append(item)
topics = topics_1

#8) clean text for topic in test df
topics_test = df1['topics'].tolist()
topics_2 = []
for i in topics_test:
    item = []
    for y in i:
        item.append(clean_text(y))
    topics_2.append(item)
topics_test = topics_2

"""**STEP 3:** Processing categorical variable: Venue, Topics, Fields of study

**Venue**
"""

#1) Venue in train df
df['venue'] = df['venue'].astype("category")
df['venue'] = df['venue'].cat.codes

#2) Venue in test_df
df1['venue'] = df1['venue'].astype("category")
df1['venue'] = df1['venue'].cat.codes

"""**Topics** removed since it decreased performance"""

# #0) read the shape of train df and test df1 -> use this index to split the df later
# print(df.shape)
# print(df1.shape)
# #1) combine 2 list of topics + topics_test
# topics_joined = topics + topics_test
# #2) stack train df and test df1 on top of each other
# pieces = (df,df1)
# df_concat = pd.concat(pieces)
# #3) impute topics_joined into concat df
# df_concat['topics'] = topics_joined
# topics_snip = df_concat['topics']
# #4) apply MultiLabelBinarier() for all topics in topics_snip df
# mlb = MultiLabelBinarizer()
# topics_df_concat = pd.DataFrame(mlb.fit_transform(topics_snip), columns = mlb.classes_)
# #5) split back the train and test df topics with multilabelbinarizer application
# topics_df_train = topics_df_concat.iloc[:9657,]
# topics_df_test = topics_df_concat.iloc[9657:,]
# topics_df_test = topics_df_test.reset_index()
# topics_df_test = topics_df_test.drop(columns=['index'])
# print(topics_df_train.shape) #print the shape of multilabelbinarizer df for train df
# print(topics_df_test.shape) #print the shape of multilabelbinarizer df for test df

"""**Fields of study** removed since it decreased performance """

# #1) extract the fields of study column from concat df + fill NaN with 0
# fields_of_study_snip = df_concat['fields_of_study']
# fields_of_study_snip = fields_of_study_snip.fillna("0")
# #2) Apply multilabel binerizer 
# fields_of_study_df_concat = pd.DataFrame(mlb.fit_transform(fields_of_study_snip), columns = mlb.classes_)
# #3) split back the train and test df with multilabelbinarizer application + drop the "0" column (meaning the column with NaN type counted)
# fields_of_study_df_train = fields_of_study_df_concat.iloc[:9657,]
# fields_of_study_df_train = fields_of_study_df_train.drop(columns=['0'])
# fields_of_study_df_test = fields_of_study_df_concat.iloc[9657:,]
# fields_of_study_df_test = fields_of_study_df_test.drop(columns=['0'])
# fields_of_study_df_test = fields_of_study_df_test.reset_index()
# fields_of_study_df_test = fields_of_study_df_test.drop(columns=['index'])
# print(fields_of_study_df_train.shape) #print the shape of multilabelbinarizer df for train df
# print(fields_of_study_df_test.shape) #print the shape of multilabelbinarizer df for test df

"""**STEP 4:** tf-idf for cleaned words in title + abstract in both train and test dataset"""

#1) feature vector title -> df
vec1 = TfidfVectorizer(min_df = 500)
vec2 = TfidfVectorizer(min_df = 0.1)
test_counts_title_vectors = vec1.fit_transform(title_combined)
title_df = pd.DataFrame(test_counts_title_vectors.todense(), columns=vec1.get_feature_names())
title_df.replace([np.inf, -np.inf], np.nan, inplace=True)
title_df.fillna(0, inplace=True)

#2) feature vector abstract -> df
test_counts_abstract_vectors = vec2.fit_transform(abstract_combined)
abstract_df = pd.DataFrame(test_counts_abstract_vectors.todense(), columns=vec2.get_feature_names())
abstract_df.replace([np.inf, -np.inf], np.nan, inplace=True)
abstract_df.fillna(0, inplace=True)

"""**STEP 5:** Generate train and test features dataset"""

#1) X for train dataset
#Feature selection: title length, abstract length, number of authors, number of topics, reference number, year, is open access, venue, title word counts, abstract word counts
feature_name = ['title_length','abstract_length','authors_number','topics_number','references','year','is_open_access','venue']

#title length
title_length = []
for i in title:
  title_length.append(len(i))
#abstract length
abstract_length = []
for i in abstract:
  abstract_length.append(len(i))
#number of authors 
authors_number = []
for i in authors:
  authors_number.append(len(i))
#number of topics 
topics_number = []
for i in topics:
  topics_number.append(len(i))

#create feature dataframe for above stated lists + references, year, is_open_acces, venue 
train1 = pd.DataFrame(list(zip(title_length,abstract_length,authors_number,topics_number,df['references'],df['year'],df['is_open_access'],df['venue'])),columns = feature_name)
train1 = pd.concat([train1,title_df,abstract_df],join='inner',axis = 1)
train1.replace([np.inf, -np.inf], np.nan, inplace=True)
train1.fillna(0, inplace=True)
# print(tabulate(train1.head(), headers='keys', tablefmt='psql')) -> to glance at the train features if needed
#feature
X_train = train1
#scaling X on numerical features to avoid skewness and other effects
scale = StandardScaler()
X_train[['title_length','abstract_length','authors_number','topics_number','references','year']] = scale.fit_transform(X_train[['title_length','abstract_length','authors_number','topics_number','references','year']])
X_train = np.array(X_train).astype(float)

#2) X for test dataset
#title length
title_length_test = []
for i in title_test:
  title_length_test.append(len(i))
#abstract length
abstract_length_test = []
for i in abstract_test:
  abstract_length_test.append(len(i))
#number of authors 
authors_number_test = []
for i in authors_test:
  authors_number_test.append(len(i))
#number of topics 
topics_number_test = []
for i in topics_test:
  topics_number_test.append(len(i))

#create feature dataframe for above stated lists + references, year, is_open_acces, venue 
test1 = pd.DataFrame(list(zip(title_length_test,abstract_length_test,authors_number_test,topics_number_test,df1['references'],df1['year'],df1['is_open_access'],df1['venue'])),columns = feature_name)
test1 = pd.concat([test1,title_df,abstract_df],join='inner',axis = 1)
test1.replace([np.inf, -np.inf], np.nan, inplace=True)
test1.fillna(0, inplace=True)
# print(tabulate(test1.head(), headers='keys', tablefmt='psql')) -> to glance at the test features if needed
#feature
X_test = test1
#scaling X on numerical features to avoid skewness and other effects
X_test[['title_length','abstract_length','authors_number','topics_number','references','year']] = scale.transform(X_test[['title_length','abstract_length','authors_number','topics_number','references','year']])
X_test = np.array(X_test).astype(float)
#print the shape of the train and test feature 
print(X_train.shape)
print(X_test.shape)

"""**STEP 6:** Setting up Y_train and Y_test"""

#Y_train reshaping + logging to lower outliers
Y_train = np.array(df['citations']).reshape(-1,1)
Y_train = np.log1p(Y_train)

"""**STEP 7:** Hyperparameter tunning to get the best parameters """
#running this takes hours, therefore we made a sperate random forest with the parameters found by the gridsearch
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 5000, num = 10)]
max_depth = [int(x) for x in np.linspace(1, 200, num = 11)]
max_features = [int(x) for x in np.linspace(start = 5, stop = 90, num = 10)]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth}

rf1 = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf1, param_distributions = random_grid, n_iter = 20, cv = 5,random_state = 42)


"""**STEP 8:** Making predictions with the best model"""

#random forest model
rf = RandomForestRegressor(n_estimators=1733, max_depth=27,max_features=30, random_state=42) #parameters from the random grid search
rf.fit(X_train,np.ravel(Y_train))
Y_test = rf.predict(X_test)

#convert back the log transformed of prediction to the original prediction
Y_test = np.expm1(Y_test)

#import google package to download files
from google.colab import files
#convert file to json format
predicted = pd.DataFrame(zip(df1['doi'],Y_test), columns = ['doi','citations'])
js = predicted.to_json("predicted.json",orient='records')
#download the file
files.download("predicted.json")