import nltk
from nltk import FreqDist
from nltk.corpus import stopwords

f = open('2.15 merge.csv','rU') #read file from local 
raw = f.read()

raw = raw.replace('\n',' ') #replace all ‘\n’ by ‘  ’(space)
raw = raw.decode('utf8') #decode raw text by utf-8

tokens = nltk.word_tokenize(raw)

#Stopwords Removal and only keep text data then change to lowercase

mystopwords = stopwords.words('english')

mystopwords.extend(['https','rt','ed','ht','htt'])
           
words = [w.lower() for w in tokens if w.isalpha() if w.lower()not in mystopwords]

wnl = nltk.WordNetLemmatizer()
stem = [wnl.lemmatize(w) for w in words]
stem = [w.encode('utf8') for w in stem]
freq = FreqDist(stem)
sorted_freq = sorted(freq.items(),key = lambda k: k[1], reverse = True)


sorted_freq

freq.plot(40)


with open('UN_tf_Lemmatizer.csv','w') as f:
    for word,frequency in sorted_freq: 
        f.write(str(word)+','+str(frequency)+'\n')