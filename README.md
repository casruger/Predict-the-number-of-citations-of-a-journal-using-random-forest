# Predict the number of citations of an article using random forest
Predict the number of citations an article will get using a random forest and, among others, features created by a Term Frequency Inverse Document Frequency Vectorizer

# Project description in short
Create a machine learning model which can predict the number of citations an article will get. The training and test data sets are provided. Both containing the following information: doi, title, field of study, abstract, year of publication, venue, authors, number of references per article, and topics. The goal is to score the lowest r^2 on the given test data. 

# How we build the model 

## Feature Engineering
Besides the provided features such as the year of publication, venue, and the number of references per article, we
generated some additional features, which will be further elaborated on in this section. First, to process the
textual data (e.g., the abstract and title) to readable data for the algorithm, we used the TfidfVectorizer
function of the scikit learn library. This function converts the textual data into a sparse matrix of token count
frequencies, which is interpretable by the algorithm. Before the usage of the TfidfVectorizer function, we
cleaned textual data by removing stop words and lower-casing them, as described by Bernardo (2019) . Besides
that, we also generated features that represent the length of the title, abstract, number of authors, and number of
topics. According to Chakraborty et. al (2014) and Yan et. al (2011), these are one of the more important
features to be able to predict the citation count. To account for the numerical features, we used the
StandardScaler function, subtracting the mean and then dividing that by the standard deviation. If this
function would not be used, the algorithm could behave badly seeing that the numerical data is on a different
scale. To process the label, we log-transformed the citation count in the train dataset using numpy.log1p
because of the big outliers. Lastly, after fitting this label and set of features in the learning algorithm, we
converted the predictions back using numpy.expm1. The whole process of data processing and applying
the learning algorithm was done using Google Colab interface. Hence, we used some packages to load our
working directory as well as download the output such as google.colab drive and google.colab
files.

## Learning Algorithm(s)
To choose the best learning algorithm we utilized a function lazypredict imported from
lazypredict.supervised. This function runs a rather large number of algorithms and provides you with the best
model in terms of R2 and RMSE. Based on this output and the lectures during the course, we decided to use
the Random Forest Regressor. This model was one of the best performing models of the Lazy Predict function
and it is a unique opportunity to immediately implement part of the course content into our project. A random
forest regression model will fit a number of decision trees on subsets of the dataset. The settings of this learning
algorithm was tuned and is discussed in the next section.

## Hyperparameter Tuning
An important step in training a machine learning algorithm is to tune the hyperparameters. In our case we
chose to use the RandomizedSearchCV of the scikit learn library. This function does a randomized search
cross validation on the hyperparameters to determine the best ones. We chose the random search function over
the grid search cross validation seeing that we have a rather large number of parameters to tune. In that case it
is recommended to use the random search cross validation. This function optimizes the hyper parameters by
drawing random values of the parameters and evaluating them. Then, the function selects the most successful
model and provides the optimal hyper parameter settings. 

## References 
Chakraborty, T., Kumar, S., Goyal, P., Ganguly, N., & Mukherjee, A. (2014). Towards a stratified
learning approach to predict future citation counts. IEEE/ACM Joint Conference on Digital
Libraries, 1–10. https://doi.org/10.1109/jcdl.2014.6970190

Yan, R., Tang, J., Liu, X., Shan, D., & Li, X. (2011). Citation count prediction: Learning to estimate
future citations for literature. Proceedings of the 20th ACM international conference on Information
and knowledge management - CIKM ’11, 1–6. https://doi.org/10.1145/2063576.2063757

Bernardo (2019). Reddit-Classifier/02 EDA.ipynb at master · berkurka/Reddit-Classifier.GitHub.
https://github.com/berkurka/Reddit-Classifier/blob/master/Notebooks/02%20EDA.ipynb

How do I get a list of all the duplicate items using pandas in python? (2013, February 2).
StackOverflow. https://stackoverflow.com/questions/14657241/how-do-i-get-a-list-of-all
-the-duplicate-items-using-pandas-in-python

Pretty Printing a pandas dataframe. (2013, August 30). Stack Overflow.
https://stackoverflow.com/questions/18528533/pretty-printing-a-pandas-dataframe

# Result
We reached an R-squared of 0.39. Meaning we could predict around 39% of the variance. 
There were some investigations regarding the usage of H-index to account for the author’s popularity, which
we could have scrapped from Elsevier’s Python API (Elsapy). However, due to time constraints, we could not
find a way to properly use this powerful package. Moreover, we did not spend enough time to apply the Sklearn
Feature selection algorithm, which might be the cause of reducing the prediction efficiency. Lastly, we would
have used a cosine similarity which might be beneficial to the model.

