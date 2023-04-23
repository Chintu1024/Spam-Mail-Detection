import streamlit as st
import pickle
import string
import nltk
import sklearn
from nltk.corpus import stopwords
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from sklearn.svm import SVC

ps = PorterStemmer()

def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  y=[]
  for i in text:
    if i.isalnum(): #this is to remove special chars
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  ps = PorterStemmer()
  for i in text:
    y.append(ps.stem(i)) #applying stemming

  return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.image('./spam-filter.png')
st.title("Email Spam Detector")
input = st.text_area("Enter the content : ")

if st.button('Predict'):
  # 1. preprocess
  transform_input = transform_text(input)
  # 2. vectorize
  vector_input = tfidf.transform([transform_input])
  # 3. predict
  result = model.predict(vector_input)[0]
  # 4. display
  if result == 1:
    st.header("Spam")
  else:
    st.header("Not Spam")

