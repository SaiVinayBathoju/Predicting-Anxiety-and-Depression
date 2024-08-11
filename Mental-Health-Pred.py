#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as py
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
import string
import nltk
import re


# In[ ]:


dataset_columns = ["target", "ids", "date", "flag", "user", "text"]
dataset_encode = "ISO-8859-1"
df=pd.read_csv("Mental-Health-Pred.csv", encoding = dataset_encode,names=dataset_columns)


# In[5]:


df.head()


# In[6]:


df.drop(['ids','date','flag','user'],axis = 1,inplace = True)


# In[7]:


df['target'].value_counts()


# In[8]:


#remove punctuation
def remove_punctuation(text):
    no_punct=[words for words in text if words not in string.punctuation]
    words_wo_punct=''.join(no_punct)
    return words_wo_punct
df['clean_text']=df['text'].apply(lambda x: remove_punctuation(x))
df.head()


# In[9]:


#remove hyperlink
df['clean_text'] = df['clean_text'].str.replace(r"http\S+", "") 
#remove emoji
df['clean_text'] = df['clean_text'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)
#convert all words to lowercase
df['clean_text'] = df['clean_text'].str.lower()
df.head()


# In[10]:


#tokenization
nltk.download('punkt')
def tokenize(text):
    split=re.split("\W+",text) 
    return split
df['clean_text_tokenize']=df['clean_text'].apply(lambda x: tokenize(x.lower()))


# In[11]:


import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

stopwords.words('english')


# In[12]:


# #stopwords
# nltk.download('stopwords')
stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text=[word for word in text if word not in stopword]
    return text
df['clean_text_tokenize_stopwords'] = df['clean_text_tokenize'].apply(lambda x: remove_stopwords(x))
df.head(10)


# In[13]:


new_df = pd.DataFrame()
new_df['text'] = df['clean_text']
new_df['label'] = df['target']
new_df['label'] = new_df['label'].replace(4,1)


# In[14]:



print(new_df.head())
print('Label: \n', new_df['label'].value_counts())


# In[15]:


from sklearn.model_selection import train_test_split
X = new_df['text']
y = new_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# In[16]:


y_train.value_counts()


# In[18]:


model = make_pipeline(TfidfVectorizer(), MultinomialNB())


# In[20]:



model.fit(X_train,y_train)


# In[21]:


validation = model.predict(X_test)


# In[22]:


validation1 = model.predict(X_train)


# In[23]:


from sklearn.metrics import accuracy_score
accuracy_score(y_train, validation1)


# In[24]:


cf_matrix = confusion_matrix(y_test, validation)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Greens')


# In[25]:


print(classification_report(y_test, validation))


# In[26]:


train = pd.DataFrame()
train['label'] = y_train
train['text'] = X_train

def predict_category(s, train=X_train, model=model):
    pred = model.predict([s])
    return pred[0]


# In[27]:


predict_category("i wanna shot myself")


# In[31]:


predict_category("i Kill you")


# In[29]:


predict_category("I'm cute")


# In[33]:


predict_category("i hate you")


# In[34]:


predict_category("i am feeling low")


# In[30]:


predict_category("I hate my self")


# In[35]:


predict_category("i want to die")


# In[37]:


predict_category("i am happy")


# In[ ]:




