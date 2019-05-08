import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import OneHotEncoder 
from utils import *
sample_size = 310

dialogue_df = pd.read_csv('data/dialogues.tsv', sep='\t').sample(sample_size, random_state=0)
questions_df = pd.read_csv('data/questions.csv', sep=',').sample(sample_size, random_state=0)

dialogue_df['text'] = [text_prepare(x) for x in dialogue_df['text']]
questions_df['title'] = [text_prepare(x) for x in questions_df['title']]

X = questions_df['scheme'].unique()
Cat = questions_df['category'].unique()
New=[]
for c in X:
    New.append(c)
for c in Cat:
    New.append(c)
Q=[text_prepare(x) for x in New]
import string
st=string.ascii_lowercase
chemd={}
for i in st:
    chemd[i]=[int(j==i) for j in st]
a=[]
for i in Q:
    for j in i.split():
        a.append(j)
a=[i for i in set(a)]
d=[]
for j in a:
    s=np.zeros(26)
    for i in j:
        s=s+chemd[i]
    d.append(s/len(j))
    
def question_to_vec_char(question):
   
    arr=question.split()
    vec = np.zeros(dim)
    i=0
    for word in arr:
        if word in embeddings:
            vec+=embeddings[word]
            i+=1
        else:
            best_scheme = np.argmax(cosine_similarity(s, d)[0])
            
            if a[best_scheme] in embeddings:
                vec=vec+embeddings[a[best_scheme]]
                i+=1
    if i!=0:
        return vec/i
    else:
        return vec
    


def predict (inp):
    X = questions_df['scheme'].unique()
    Cat = questions_df['category'].unique()
    New=[]
    for c in X:
        New.append(c)
    for c in Cat:
        New.append(c)
    Q=[text_prepare(x) for x in New]
    a=[]
    for i in Q:
        for j in i.split():
            a.append(j)
    a=[i for i in set(a)]
    b={}
    for i in range(len(a)):
        b[a[i]]=[int(i==j) for j in range(len(a))]
    R=[]
    for sch in Q:
        arr=sch.split()
        vec = np.zeros(len(a))
        i=0
        for word in arr:
            vec=vec+b[word]
            i+=1

        if i!=0:
            vec= vec/i
        R.append(vec)
    W=np.zeros(len(a))
    for i in inp.split():
        if i in b:
            W+=b[i]
    W=W.reshape(1,-1)
    best_idx = np.argmax(cosine_similarity(W, R)[0])
    return (Q[best_idx])

