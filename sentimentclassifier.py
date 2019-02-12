import glob
import pandas as pd
pd.options.mode.chained_assignment = None
from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()
import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
LabeledSentence = gensim.models.doc2vec.LabeledSentence #
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from sklearn.preprocessing import scale
import pickle
from tweet_tokenizer import CustomTokenizer
n=1000000; n_dim = 200
tweet_tokenizer = CustomTokenizer()
from keras.utils import to_categorical
from collections import Counter
from tensorflow.python.client import device_lib

dimension_w2v = 210

def tokenize(tweet):
    try:
        tweet = (tweet.lower())
        tokens = tokenizer.tokenize(tweet)
        # Remove tokens starting with @ and #
        tokens = filter(lambda t: not t.startswith('@'), tokens)
        tokens = filter(lambda t: not t.startswith('#'), tokens)
        tokens = list(filter(lambda t: not t.startswith('http'), tokens))
        return tokens
    except:
        return 'NC'

#  http://help.sentiment140.com/for-students/
# Massive Positive /negative dataset ~ 160 000  long
def getDatData():
    df = pd.read_csv('/home/henkdetank/PycharmProjects/TextMining/Data/training.1600000.processed.noemoticon.csv', sep=',', error_bad_lines=False,encoding='latin-1')
    df.drop(df.columns[[1, 2, 3, 4]],axis=1,inplace=True)
    df.columns = ['sentiment', 'tweet']
    df['sentiment'] = df['sentiment'].map({4: 1, 0: 0})

    print('DatData met duplicates ', df.shape)
    df.drop_duplicates('tweet')
    print('DatData zonder duplicates ', df.shape)

    return df['sentiment'], df['tweet']


#  All SemEvalData 2013 - 2016  ~ 52 000 long
def getAllSemTaskAData():
    dfEmpty = pd.DataFrame()
    path = "/home/henkdetank/PycharmProjects/TextMining/TaskATextMining/*"
    files = glob.glob(path)
    frames = []
    for name in files:
        dtemp = pd.read_csv(name, sep='\t', error_bad_lines=False)
        dtemp.columns = ['id', 'sentiment', 'tweet']
        frames.append(dtemp)

    df = pd.concat(frames)


    df['sentiment'] = df['sentiment'].replace(['objective'], 0)
    df['sentiment'] = df['sentiment'].replace(['neutral'], 0)
    df['sentiment'] = df['sentiment'].replace(['positive'], 1)
    df['sentiment'] = df['sentiment'].replace(['negative'], -1)
    df.drop(['id'], axis=1, inplace=True)

    filtered_df = df[df.isnull()]

    print(df.shape)
    df.drop([11063], inplace=True)
    print(df.shape)



    return df['sentiment'], df['tweet']


def load_obj(name):
        with open('obj/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

def save_obj(obj, name):
        with open('obj/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def train_w2v(X_train_SemEvalA):

    tweet_w2v = Word2Vec(size=dimension_w2v, min_count=1)
    tweet_w2v.build_vocab([word for word in X_train_SemEvalA])
    tweet_w2v.train(tqdm([word for word in X_train_SemEvalA]), total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)

    print("Build a Tf-Id matrix")
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=5)
    matrix = vectorizer.fit_transform([x for x in X_train_SemEvalA])
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    print('vocab size :', len(tfidf))

    save_obj(tweet_w2v, "tweet_w2v")
    save_obj(tfidf, "tfidf")





def main():


    y_semeval, x_semeval = getAllSemTaskAData()
    y_emoticon, x_emoticon = getDatData()

    X = pd.concat([x_semeval, x_emoticon])
    y = pd.concat([y_semeval, y_emoticon])
    X = ([tokenize(tweet) for tweet in X])

    X_train_SemEvalA, X_test_SemEvalA, y_train_SemEvalA, y_test_SemEvalA = train_test_split(X, y, test_size=0.20, random_state=42)

    save_obj(X_train_SemEvalA, "X_train")
    save_obj(X_test_SemEvalA, "X_test")
    save_obj(y_train_SemEvalA, "y_train")
    save_obj(y_test_SemEvalA, "y_test")

    print("x test semevalA A  " , len(X_test_SemEvalA))
    print("Y test semeval A ", len(y_test_SemEvalA))


    tweet_w2v = load_obj("tweet_w2v")
    tfidf = load_obj("tfidf")

    def tweet_tfidf__w2vec_vector(tokens, size):
        vec = np.zeros(size).reshape((1, size))
        count = 0.
        for word in tokens:
            try:
                vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
                count += 1.
            except KeyError:
                # handling the case where the token is not
                # in the corpus. useful for testing.
                continue
        if count != 0:
            vec /= count
        return vec

    print("waar gaat")

    train_vecs_w2v = np.concatenate(([tweet_tfidf__w2vec_vector(z, dimension_w2v) for z in tqdm(map(lambda x: x, X_train_SemEvalA))]))
    print("dit fout hier")
    train_vecs_w2v = scale(train_vecs_w2v)
    print('of hier')
    save_obj(train_vecs_w2v,"train_vecs_w2v")

    print("niet hier ")


    test_vecs_w2v = np.concatenate(([tweet_tfidf__w2vec_vector(z, dimension_w2v) for z in tqdm(map(lambda x: x, X_test_SemEvalA))]))
    test_vecs_w2v = scale(test_vecs_w2v)
    save_obj(test_vecs_w2v,"test_vecs_w2v")


    train_vecs_w2v = load_obj("train_vecs_w2v")
    test_vecs_w2v = load_obj('test_vecs_w2v')

    dimension_w2v = 200
    y_train_SemEvalA = load_obj("y_train")
    y_test_SemEvalA = load_obj("y_test")

    x, y = getDatData()
    X, Y = getAllSemTaskAData()

    y_train_SemEvalA = to_categorical(y_train_SemEvalA, num_classes=3)
    y_test_SemEvalA =  to_categorical(y_test_SemEvalA, num_classes=3)

    print(y_train_SemEvalA.shape)
    print(y_test_SemEvalA.shape)

    print("trololo gpu", device_lib.list_local_devices())

    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=dimension_w2v))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_vecs_w2v, y_train_SemEvalA, epochs=5, batch_size=32, verbose=2)
    y_values = model.predict(train_vecs_w2v, y_train_SemEvalA, epochs=5, batch_size=32, verbose=2)


    print(y_values)

    score = model.evaluate(test_vecs_w2v, y_test_SemEvalA, batch_size=128, verbose=2)
    print(score[1])



def fitthamodel():
    train_vecs_w2v = load_obj("train_vecs_w2v")
    test_vecs_w2v = load_obj('test_vecs_w2v')

    dimension_w2v = 200
    y_train  = load_obj("y_train")
    y_test_SemEvalA = load_obj("y_test")


    x, y = getDatData()
    X, Y = getAllSemTaskAData()


    y_train = to_categorical(y_train, num_classes=3)
    y_test_SemEvalA =  to_categorical(y_test_SemEvalA, num_classes=3)
    print(y_train)
    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=dimension_w2v))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_vecs_w2v, y_train, epochs=8, batch_size=32, verbose=2)


    score = model.evaluate(test_vecs_w2v, y_test_SemEvalA, batch_size=128, verbose=2)
    print(score[1])

if __name__ == '__main__':
    main()
    # fitthamodel()

'''
epoch 1/8
- 35s - loss: nan - acc: 0.7311
Epoch 2/8
 - 35s - loss: nan - acc: 0.0000e+00
Epoch 3/8
 - 37s - loss: nan - acc: 0.0000e+00
Epoch 4/8
 - 34s - loss: nan - acc: 0.0000e+00

    ya, X1 = getAllSemTaskAData()
    yemoticon, Xemoticon = getDatData()

    framesy = [ya, yemoticon]
    framesx = [X1, Xemoticon]

    X = pd.concat(framesx)
    y = pd.concat(framesy)

    full = pd.concat([y,X], axis=1)

    print(X.shape)
    full_X = full.drop_duplicates()
    print(X.shape)
    full_X.columns = ['sentiment', 'tweet']
    X = full_X['tweet']
    y = full_X['sentiment']

    X = ([tokenize(tweet) for tweet in X])

    X_train_SemEvalA, X_test_SemEvalA, y_train_SemEvalA, y_test_SemEvalA = train_test_split(X1, ya, test_size=0.33, random_state=42)
    X_train = X
    y_train = y


    print("y : ", y_train.shape)
    print("X: ", len(X_train))
    print("x test semevalA A  " , len(X_test_SemEvalA))
    print("Y test semeval A ", len(y_test_SemEvalA))



    dimemsion_w2v = 210


    tweet_w2v = Word2Vec(size=dimemsion_w2v, min_count=2)
    tweet_w2v.build_vocab([word for word in X_train])
    tweet_w2v.train(tqdm([word for word in X_train]), total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)

    print("Build a Tf-Id matrix")
    vectorizer = TfidfVectorizer(   analyzer=lambda x: x, min_df=5)
    matrix = vectorizer.fit_transform([x for x in X_train])
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    print('vocab size :', len(tfidf))


    # TODO: Open the file to save as pkl file
    word2vec_trained = open("word2vec.pkl", 'wb')
    pickle.dump(tweet_w2v, word2vec_trained)
    # TODO: Close the pickle instances
    word2vec_trained.close()
    word2vec_model_pkl = open("word2vec.pkl", 'rb')
    tweet_w2v = pickle.load(word2vec_model_pkl)

    def tweet_tfidf__w2vec_vector(tokens, size):
        vec = np.zeros(size).reshape((1, size))
        count = 0.
        for word in tokens:
            try:
                vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
                count += 1.
            except KeyError:  # handling the case where the token is not
                # in the corpus. useful for testing.
                continue
        if count != 0:
            vec /= count
        return vec

    train_vecs_w2v = np.concatenate(([tweet_tfidf__w2vec_vector(z, dimemsion_w2v) for z in tqdm(map(lambda x: x, X_train))]))
    train_vecs_w2v = scale(train_vecs_w2v)
    # np.save('train_vecs',train_vecs_w2v)

    test_vecs_w2v = np.concatenate(([tweet_tfidf__w2vec_vector(z, dimemsion_w2v) for z in tqdm(map(lambda x: x, X_test_SemEvalA))]))
    test_vecs_w2v = scale(test_vecs_w2v)
    # np.save('test_vecs',test_vecs_w2v)



    # train_vecs_w2v = np.load('train_vecs.npy')
    # test_vecs_w2v = np.load('test_vecs.npy')

    # loss = 'categorical_crossentropy' and activation = 'softmax'

    # y_binary = to_categorical(y_test_SemEvalA)

    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=dimemsion_w2v))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])




    model.fit(train_vecs_w2v, y_train, epochs=8, batch_size=32, verbose=2)

    score = model.evaluate(test_vecs_w2v, y_test_SemEvalA, batch_size=128, verbose=2)
    print(score[1])
'''