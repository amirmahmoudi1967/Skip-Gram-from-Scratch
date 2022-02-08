from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize

# added imports
import nltk
from nltk.tokenize import word_tokenize

__authors__ = ['Chloe Daems','Anne-Claire Laisney','Amir Mahmoudi']
__emails__  = ['chloe.daems@student-cs.fr','anneclaire.laisney@student-cs.fr','amir.mahmoudi@student-cs.fr']

# ----------------------------------
# Tokenization 

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])

def word_repetition(text):
    counter = {}
    for sentence in text:
        for word in sentence:
            if word in counter:
                counter[word] += 1
            else:
                counter[word] = 1
    return counter

def text2sentences(path):
    # feel free to make a better tokenization/pre-processing
    # First, we want to remove all special characters and numbers
    sentences = []
    punctuation = ['#','|','?','_','!','€',':',']','^',
                    '/','*','}','$','~','£','\n','-','&',
                    '{','[','(','"','`','=','','@','+',
                    '%',')','>','.',',','\\','"']
    digits = [ 0,1,2,3,4,5,6,7,8,9 ]
    
    with open(path, encoding='utf8') as f:
        for l in f:
            # removing punctuation
            for p in punctuation:
                l = l.replace(p,'')
            # removing digits
            for d in digits:
                l = l.replace(str(d),'')
            # stemming
            l = lemmatize_text(l)
            # lowercase + splitting to create unigrams
            l = l.lower().split()
            #remove empty words 
            l = [x.strip() for x in l if len(x.strip())>1]
            
            sentences.append(l)
    return sentences

# ----------------------------------

def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs

# ----------------------------------

class SkipGram:
    
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5, epochs = 10, learningRate = 0.1):
        
        self.w2id = {} # word to ID mapping
        self.trainset = sentences # set of sentences
        self.vocab = {} # list of valid words
        
        self.nEmbed = nEmbed # size of the embeddings
        self.negativeRate = negativeRate # number of noisy example for each context word
        self.winSize = winSize # size of the context window
        self.minCount = minCount # minimal recurrency of a word to be kept in the vocabulary
        
        # Training parameters
        self.epochs = epochs
        self.learningRate = learningRate
        
        # Loss and support variables
        self.loss = []
        self.accLoss = 0
        self.trainWords = 0
        
        #Context size: weight more closer context
        counter = word_repetition(self.trainset)
        
        for word in counter:
            if counter[word] >= self.minCount:
                self.vocab[word] = counter[word]
        
        #For ns(x), use p(w) = #w^(3/4)/sum(#w'^(3/4))
        # Nominator of the fraction p(w)
        for id_,w in enumerate(self.vocab):
            self.w2id[w] = id_
            self.vocab[w] = self.vocab[w] ** (3/4)
            
        sum_ = 0
        for w in self.vocab.values():
            sum_ += w
        # Fraction
        for w in self.vocab:
            self.vocab[w] = self.vocab[w]/sum_
            
        # Initializes weights to random normal distributed numbers
        # Two embeddings for each word (one as context, one as word) 
        # Might want to initialize them differently
        self.wEmbed = np.random.random( size=(len(self.vocab.values()), self.nEmbed) ) * 0.2
        self.cEmbed = np.random.random( size=(len(self.vocab.values()), self.nEmbed) ) * 0.1
        
        # Initialize Unigram Table
         # http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
        # Size of Unigram Table: 100 millions
        self.unigramTableSize = int(1e8)
        
        # fill this table with the index of each word in the vocabulary multiple times, and the number of times a word’s index appears in the table is given by P(wi) * table_size
        #Create the unigram table
        
    # ----------------------------------
    
    def sample(self, omit):
        """
        Samples negative words, ommitting those in set omit
        """
        #  we construct k samples (w,c1),...,(w,ck), where each cj is drawn according to its unigram distribution raised to the 3/4 power
        
        samples_negative_words = []
        
        while len(random_sample_id) < self.negativeRate:
            sample = self.unigramTable[np.random.randint(0, self.unigramSize)]
            if sample not in omit:
                samples_negative_words.append(sample)
        
        return samples_negative_words

    # ----------------------------------
    
    # σ(x) = 1/(1+exp(-x))
    def sigma(self,x):
        '''
        Returns the value of sigma(x)
        '''
        return 1/(1 + np.exp(-x))
    
    def gradient_(x,y):
        return (x-1) * y
    
    # Update both embeddings in each iteration. 
    # Derivation is the same (symmetric), but with respect to a different variable
    # Our goal is to find parameters to maximize the probabilities that all of the observations indeed came from the data
    def train(self):
        '''
        Training function
        '''
        for epoch in range(self.epochs):
            for counter, sentence in enumerate(self.trainset):
                sentence = list(filter(lambda word: word in self.vocab, sentence))

                for wpos, word in enumerate(sentence):
                    wIdx = self.w2id[word]
                    winsize = np.random.randint(self.winSize) + 1
                    start = max(0, wpos - winsize)
                    end = min(wpos + winsize + 1, len(sentence))
                    
                    for context_word in sentence[start:end]:
                        ctxtId = self.w2id[context_word]
                        ctxtId = 0
                        if ctxtId == wIdx: continue
                        negativeIds = self.sample({wIdx, ctxtId})
                        self.trainWord(wIdx, ctxtId, negativeIds)
                        self.trainWords += 1

                if counter % 1000 == 0:
                    print (" > training %d of %d" % (counter, len(self.trainset)))
                    self.loss.append(self.accLoss / self.trainWords)
                    self.trainWords = 0
                    self.accLoss = 0.
            print(' > training Epoch {}: Loss = {}'.format(epoch+1,self.accLoss))
            
    def trainWord(self, wordId, contextId, negativeIds):
        '''
        calculating gradients and Update both embeddings in each iteration
        '''
        #  v_w and v_c ∈ Rd are vector representations for w and c respectively
        v_w = self.wEmbed[wordId]
        v_c = self.cEmbed[contextId]
        
        # Backpropagation
        # We want to calculate argmax(sum_1(f) + sum_2(g))
        
        # sum_1 of the pair (w,c) on the set D which is all word and context pairs we extract from the text
        # f = log( sigma( v_c . v_w ) )  
        f = sigma(v_c.dot(v_w))
        gradient_v_w = gradient_(f,v_w)
        grad_v_c = gradient_(f,v_c)
        
        # sum_2 of the pair (w,c) on the set D′ of randomly sampled negative examples
        # where g = log( sigma(-v_c . v_w ) )
        
        for negativeId in negativeIds:
            v_c_neg = self.cEmbed[negativeId]
            g -= np.log(sigma(-v_w.dot(v_c_neg)))
            gradient_v_c_neg = gradient_(g,v_c_neg)
            gradient_v_w += gradient_(g,v_c_neg)
            self.cEmbed[negativeId] -= self.learningRate * gradient_c_neg
        
        self.wEmbed[wordId] -= self.learningRate * gradient_v_w
        self.cEmbed[contextId] -= self.learningRate * grad_v_c
        self.accLoss += loss
        
    # ----------------------------------
    
    def save(self,path):
        """
        save data into the file
        """
        skipGramData = {
            'w2id': self.w2id,
            'trainset': self.trainset,
            'vocab': self.vocab,
            'nEmbed': self.nEmbed,
            'negativeRate': self.negativeRate,
            'winSize': self.winSize,
            'minCounts': self.minCounts,
            'epochs': self.epochs,
            'learningRate': self.learningRate,
            'loss': self.loss,
            'accLoss': self.accLoss,
            'trainWords': self.accLoss,
            'loss': self.loss,
            'cEmbed': self.cEmbed,
            'wEmbed': self.wEmbed,
        }
        with open(path, 'wb') as f:
            pickle.dump(skipGramData, f, pickle.HIGHEST_PROTOCOL)

    # ----------------------------------
    
    def similarity(self,word1,word2):
        """
        computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        counterA = Counter(word1)
        counterB = Counter(word2)
    
        terms = set(counterA).union(counterB)
        dotprod = sum(counterA.get(k, 0) * counterB.get(k, 0) for k in terms)
        magA = math.sqrt(sum(counterA.get(k, 0)**2 for k in terms))
        magB = math.sqrt(sum(counterB.get(k, 0)**2 for k in terms))
        return dotprod / (magA * magB)
        
    def compare_two_tokenized_sentences(first_tokenized_sentence, second_tokenized_sentence):
        """
        Cosine
        """
        return similarity(first_tokenized_sentence, second_tokenized_sentence)

    # ----------------------------------
    
    @staticmethod
    def load(path):
        with open(path, 'rb') as handle:
            return pickle.load(handle) 
        
# ----------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences)
        sg.train(...)
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = SkipGram.load(opts.model)
        for a,b,_ in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
            print(sg.similarity(a,b))

