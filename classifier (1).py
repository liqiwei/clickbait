import pandas as pd
import csv
df = pd.read_csv(r'/Users/qiweili/Desktop/other_clickbait/all_data.csv')

data = df.headline
target = df.is_clickbait


import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(    data, target, test_size=0.20, random_state=42)


from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

#what would happen with IDF=true?
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
X_train_tf = tfidf_transformer.fit_transform(X_train_counts)


# In[17]:


from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss='hinge', penalty='l2'    ,alpha=1e-3, random_state=42,max_iter=5, tol=None).fit(X_train_tf, y_train)


# In[18]:


from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect',CountVectorizer()),
                     ('tfidf',TfidfTransformer()),
                    ('clf',SGDClassifier()),
                    ])


# In[19]:


text_clf.fit(X_train,y_train)


# In[20]:


import numpy as np 
predicted = text_clf.predict(X_test)


# In[21]:


np.mean(predicted==y_test)


# In[22]:


from sklearn import metrics
metrics.confusion_matrix(y_test,predicted)


# In[23]:


#get precision recall score
y_score = text_clf.decision_function(X_test)

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.7f}'.format(      average_precision))


# using headlines I collected

# In[24]:


from sklearn.metrics import classification_report 
target_names=['clickbait', 'nonclickbait']
classification_report(y_test, predicted, target_names=target_names)


# In[25]:


#loop through the entire folder to read csv 


# In[49]:


import glob
import os
path = '/Users/qiweili/Desktop/other_clickbait/full_data'
extension = 'csv'

os.chdir(path)
results = [i for i in glob.glob('*.{}'.format(extension))]


# In[52]:


#read every result csv in the file
for result in results: 
    file = pd.read_csv(r'/Users/qiweili/Desktop/other_clickbait/full_data/'+result)
    X_test = file.title
    source_predicted = text_clf.predict(X_test)
    tmp = []
    for prediction in source_predicted:
        file['is_clickbait'] = 0
        tmp.append(prediction)
        #file['is_clickbait']=tmp
    file.to_csv('z_'+result,sep=',')



