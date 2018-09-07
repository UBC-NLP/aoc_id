from keras.preprocessing import text, sequence
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import text_normalization as tn
import gensim
import numpy as np
import pandas as pd
import io
#################################
##########################
EmbeddingSize=300
vocab_size=50000
charset = "ا ب ت ث ج ح خ د ذ ر ز س ش ص ض ع غ ف ق ك ل م و ه ي ى ط ظ ن"
other_symbols = "0123456789-,;.!:/\\@#%^*+-=()[] "
alphabet = (list(charset) + list(other_symbols) + ['\n'])
## Defining Data processing functions

def LoadData(Corpus,ClassesDict,Arabic=False): ## loading file
    DF=pd.read_csv(Corpus,converters={'text': str})
    labels=[ClassesDict[x] for x in DF['label'].tolist()]
    sentences=DF['text'].tolist()
    if Arabic:
       sentences=[tn.NormForWord2Vec(line) for line in sentences]
    TrueLabels=labels
    labels=to_categorical(np.asarray(labels), num_classes=len(ClassesDict))
    return sentences,labels,TrueLabels

def tokenizeData(X_train,X_valid,vocab_size,X_test=None): ## tokenization
    "tokenize data"
    #init tokenizer
    tokenizer= Tokenizer(nb_words=vocab_size, filters='\t\n',split=" ",char_level=False)
    #use tokenizer to split vocab and index them
    tokenizer.fit_on_texts(X_train)
    ##txt to seq
    X_train= tokenizer.texts_to_sequences(X_train)
    X_valid = tokenizer.texts_to_sequences(X_valid)
    if X_test != None:
       X_test=tokenizer.texts_to_sequences(X_test)
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    return X_train,X_valid,X_test,reverse_word_map

def paddingSequence(X_train,X_valid,maxLen,X_test=None): ## Sequence Padding
    "make sure that a sequence is satisfied max_length condition"
    #######equalize list of seq
    X_train= pad_sequences(X_train, maxLen, padding='post', truncating='post')
    X_valid= pad_sequences(X_valid, maxLen, padding='post', truncating='post')
    if X_test != None:
       X_test= pad_sequences(X_test, maxLen, padding='post',truncating='post')
    return X_train,X_valid,X_test

def read_labels(categorical=False):
    count=0
    classes={}
    for line in open("conf/label_list"):
        if line.strip("\n") not in classes:
            classes[line.strip("\n")] = count
            count += 1
    return classes
#######################

## Defining External Embedding Gensim functions

def GetEmbeddingWeights(embedding_dim,n_symbols,wordmap,vecDic):
    embedding_weights = np.zeros((n_symbols, embedding_dim))
    for index,word  in wordmap.items():
        if word in vecDic:
           embedding_weights[index, :] = vecDic[word]
        else:
           ## if doesn't exist initialize embedding vector from a random distribution
           embedding_weights[index, :] = np.random.randn(embedding_dim)
    return embedding_weights

def GetVecDicFromGensim(GensimFile):
    Model=gensim.models.Word2Vec.load(GensimFile)
    return Model.wv

def load_fasttext(FastTextFile):
    fin = io.open(FastTextFile, 'r', encoding='utf-8', newline='\n', errors='ignore')
    #n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array([list(map(float, tokens[1:]))])
    return data

#####################################
def get_char2idx(x):
    st=""
    for line in x:
        st+=line
    charset=set(st)
    vocabsize=len(charset)
    char2idx= dict(zip(list(charset),range(1,len(charset)+1)))
    return char2idx,vocabsize
#####################################
def encode_all_data2chars(x,char2idx):
    X_return=[]
    for line in x:
        idxs= [char2idx.get(c,1) for c in line]
        X_return.append(idxs)
    return X_return
#####################################
def encode_data2char(x):
    char2idx = dict(zip(list(alphabet), range(2, len(alphabet) + 2)))
    print(len(char2idx))
    X=[]
    for line in x:
        indices = [char2idx.get(c, 1) for c in line]
        X.append(indices)
    return X
#####################################
def get_word_map_num_symbols(corpus):
    X_train,Y_train,Y_Train_true=LoadData(corpus,ClassesDict=get_classes())
    X_test,Y_test,Y_test_true=LoadData(corpus,ClassesDict=get_classes())
    print ('---- Tokenizing Training and Testing Data ------')
    X_train,X_test,dummy,wordmap=tokenizeData(X_train,X_test,vocab_size=get_vocab_size())
    n_symbols=len(wordmap)+1
    return n_symbols,wordmap
#####################################
def set_corpus(Corpus):
    corpus=Corpus
    return corpus
#####################################
def get_classes():
    return Classes
def get_corpus():
    return Corpus
def get_testset():
    return Testset
def get_vocab_size():
    return vocab_size
def get_n_symbold():
    return
###########
Classes=read_labels()
#############################################
Corpus="./data/train/MultiTrain.Shuffled.csv"
Testset="./data/dev/MultiDev.csv"
