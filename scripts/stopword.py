import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

file = open("vocab-1", "rb")
output = open("vocab", "w")
for i in file:
    word = i.strip()
    if word not in stop_words:
        output.write(word+"\n")
output.close()
file.close()
