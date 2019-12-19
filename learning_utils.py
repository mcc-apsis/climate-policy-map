import re, string
import string

from nltk import wordpunct_tokenize
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords as sw
punct = set(string.punctuation)
from nltk.corpus import wordnet as wn
from sklearn.metrics import precision_score, recall_score, r2_score

from sklearn.ensemble import IsolationForest

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.animation
import matplotlib

stopwords = set(sw.words('english'))

def lemmatize(token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)
        return WordNetLemmatizer().lemmatize(token, tag)

def tokenize(X):
    for sent in sent_tokenize(X):
        for token, tag in pos_tag(wordpunct_tokenize(sent)):
            token = token.lower().strip()
            if token in stopwords:
                continue
            if all(char in punct for char in token):
                continue
            if len(token) < 3:
                continue
            if all(char in string.digits for char in token):
                continue
            lemma = lemmatize(token,tag)
            yield lemma

def plot_model_accuracy(model,x_test,y_test,ax,threshold=0.1,inv=False):
    try:
        y_prob = model.predict_proba(x_test)
        prob_y_true = y_prob[:,1]
        prob_y_false = y_prob[:,0]
    except:
        y_prob = model.decision_function(x_test)
        prob_y_true = y_prob
        prob_y_false = None

    order = np.argsort(prob_y_true)
    ordered_prob = prob_y_true[order]

    cutoff = np.argmax(ordered_prob>threshold)

    y_predicted = np.where(prob_y_true > threshold,1,0)

    from sklearn.metrics import precision_score, recall_score
    p = precision_score(y_test,y_predicted)
    r = recall_score(y_test,y_predicted)

    savings = len(y_predicted[y_predicted < threshold])

    #print("avoided checking {} out of {} documents".format(savings,len(y_predicted)))

    #print("precision = {}".format(p))
    #print("recall = {}".format(r))

    ax.scatter(
        np.arange(len(prob_y_true)),
        prob_y_true[order],
        s=2
    )
    ax.scatter(
        np.arange(len(prob_y_true)),
        y_test.reset_index(drop=True)[order],
        s=2
    )
    if inv:
        prob_y_false = prob_y_false[order]
        ax.scatter(
            np.arange(len(prob_y_true)),
            prob_y_false,
            s=2
        )
    ax.set_title("avoided={:0.2f}\nprecision={:0.2f}\nrecall={:0.2f}".format(
        savings/len(y_predicted),
        p,r
    ))
    ax.axvline(cutoff)


def precision_recall_plot(model,x_test,y_test, ax, frac):
    try:
        y_score = model.decision_function(x_test)
    except:
        y_score = model.predict_proba(x_test)[:,1]
    average_precision = average_precision_score(y_test, y_score)

    print('Average precision-recall score: {0:0.2f}'.format(
          average_precision))

    precision, recall, _ = precision_recall_curve(y_test, y_score)

    ax.step(recall, precision, color='b', alpha=0.2,
             where='post')
    ax.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title('frac={0:0.2f}\nAP={1:0.2f}'.format(
        frac,average_precision
    ))

def traintest(df,f):
    train = df.sample(frac=f)
    test = df[~df['id'].isin(train['id'])]
    return train, test

def adj_r2_score(model,y,yhat):
    """Adjusted R square — put fitted linear model, y value, estimated y value in order

    Example:
    In [142]: metrics.r2_score(diabetes_y_train,yhat)
    Out[142]: 0.51222621477934993

    In [144]: adj_r2_score(lm,diabetes_y_train,yhat)
    Out[144]: 0.50035823946984515"""
    try:
        xlen=len(model.coef_)
    except:
        xlen=model.n_support_[0]
    from sklearn import metrics
    adj = 1 - float(len(y)-1)/(len(y)-xlen-1)*(1 - metrics.r2_score(y,yhat))
    return adj

class ScreenSimulation:
    def __init__(self,df,model,X,y):
        self.partial=False
        self.df = df
        self.model=model
        self.X=X
        self.y=y
        self.train=df.sample(frac=0)
        self.test=self.df[~self.df['id'].isin(self.train['id'])].copy()
        self.test['prob'] = 0
        self.frac = 0
        self.threshold_passed = False
        self.r100 = False
        clf = IsolationForest()
        clf.fit(self.X)
        self.df['outlying'] = clf.predict(self.X)
        self.test['outlying'] = clf.predict(self.X)

    def sort_the_documents(self,strategy):
        if strategy=="outliers_first":
            if self.frac<0.1:
                sort_docs=self.test.copy().sort_values('outlying').reset_index(drop=True)
            # elif self.frac<0.2:
            #     sort_docs=self.test.copy().sample(frac=1).reset_index(drop=True)
            else:
                sort_docs=self.test.copy().sort_values('prob',ascending=False).reset_index(drop=True)
        if strategy=="time":
            sort_docs=self.test.copy().sort_values('rated').reset_index(drop=True)
        elif strategy=="relevant_first":
            sort_docs=self.test.copy().sort_values('prob',ascending=False).reset_index(drop=True)
        elif strategy=="relevant_last":
            sort_docs=self.test.copy().sort_values('prob',ascending=True).reset_index(drop=True)
        elif strategy=="relevant_first_delay":
            if self.frac<0.1:
                sort_docs=self.test.copy().sort_values('rated').reset_index(drop=True)
            else:
                sort_docs=self.test.copy().sort_values('prob',ascending=False).reset_index(drop=True)
        return sort_docs

    def update(self,i,strategy,threshold):
        #Use strategy to sort in whatever way
        sort_docs=self.sort_the_documents(strategy)
        new_docs = sort_docs.loc[sort_docs.index.intersection(self.s_docs),:]
        doc_ids = list(self.train['id'])+list(new_docs['id'])
        self.train = self.df[self.df['id'].isin(doc_ids)]
        self.test = self.df[~self.df['id'].isin(self.train['id'])].copy()
        if self.partial:
            self.model.partial_fit(
                self.X[self.train.index],
                self.y[self.train.index],
                classes=np.array([0,1])
            )
        else:
            self.model.fit(
                self.X[self.train.index],
                self.y[self.train.index]
            )

        y_prob = self.model.predict_proba(self.X[self.test.index])[:,1]
        y_predicted = np.where(y_prob > 0.05,1,0)
        y_test = self.y[self.test.index]
        y_train = self.y[self.train.index]
        y_train_prob = self.model.predict_proba(self.X[self.train.index])[:,1]

        all_trues = len(np.where(self.y==1)[0])
        trues_seen =len(np.where(self.y[self.train.index]==1)[0])

        self.test['prob'] = y_prob

        p = precision_score(y_test,y_predicted)
        r = recall_score(y_test,y_predicted)
        #r2 = adj_r2_score(self.model,y_train,y_train_prob)
        relevant_seen = trues_seen/all_trues

        self.frac = self.train.index.size/self.df.index.size

        if relevant_seen > threshold and self.threshold_passed is False:
            self.ax.text(
               1.05,0.8,
               "Threshold {} \npassed after {:.0%}".format(threshold,self.frac)
            )
            self.threshold_passed=True

        if relevant_seen==1 and self.r100 is False:
           self.ax.text(
               1.05,0.65,
               "100% recall\nafter {:.0%}".format(self.frac)
           )
           self.r100=True



        self.x.append(self.frac)

        self.y1.append(p)
        self.y2.append(r)
        #self.y3.append(r2)
        self.y4.append(relevant_seen)

        self.points1.set_offsets(
            np.c_[self.x,self.y1]
        )
        self.points2.set_offsets(
            np.c_[self.x,self.y2]
        )
        # self.points3.set_offsets(
        #     np.c_[self.x,self.y3]
        # )
        self.points4.set_offsets(
            np.c_[self.x,self.y4]
        )

    def simulate(self,iterations=25,strategy="time",threshold=0.95):
        s_size = int(np.ceil(self.df.index.size/iterations))
        self.s_docs = list(range(0,s_size))
        self.fig, self.ax = plt.subplots(dpi=100)

        self.x=[]
        self.y1=[]
        self.y2=[]
        self.y3=[]
        self.y4=[]

        dotsize = 6

        self.points1 = self.ax.scatter(0, 0.5,label="precision",s=dotsize)
        self.points2 = self.ax.scatter(0, 0.5,label="recall",s=dotsize)
        #self.points3 = self.ax.scatter(0, 0.5,label="r2",s=dotsize)
        self.points4 = self.ax.scatter(0, 0.5,label="Relevant seen",s=dotsize)

        self.ax.legend(loc="center left",bbox_to_anchor=(1, 0.5))

        self.ax.set_xlim(0,1)
        self.ax.set_ylim(0,1)

        self.ax.plot([0,1],[0,1],linestyle="--",color="grey",linewidth=1)

        self.ax.axhline(threshold,linestyle="--",color="grey",linewidth=1)


        self.ani = matplotlib.animation.FuncAnimation(
            self.fig,self.update,
            frames=iterations,repeat=False,
            fargs=(strategy,threshold,)
        )
        self.fig.tight_layout()
        plt.show()
        return
