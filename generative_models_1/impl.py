import pprint 

pp = pprint.PrettyPrinter(depth=6)

training = {}
vocabulary = set()
    
def train(features, targets):

    target_count = len(targets)
    print("Total training examples: ")+str(target_count)
    
    #add words to vocabulary set
    [vocabulary.add(word) for feature in features for word in feature]
    print "Vocabulary: "+str(vocabulary)
    
    #preprocessing -- need these constants before calculating posteri
    for i in range(target_count):
        feature = features[i]
        target = targets[i]
        
        #setup the internal data structure
        if(target not in training):
            training[target]={'count':0, 'prob': 0.0, 'blob': [], 'n': 0, 'words':{}}
            
        training[target]['blob']+=feature
        training[target]['n'] = len(list(set(training[target]['blob'])))

        #calc. priori probabilities
        training[target]['count']+=1
        training[target]['prob']= float(training[target]['count']) / float(len(targets))
    
    for i in range(target_count):
        feature = features[i]
        target = targets[i]
        
        m=len(vocabulary)
        n = training[target]['n']
        norm = n+m
 
        for word in vocabulary:
            if(word not in training[target]['words']):
                # prob should just be set to 0.0 here but i'm trying to 
                #account for words not in this example.
                training[target]['words'][word]={'count':0, 'prob':1.0/norm}
                
            if(word in feature):
                training[target]['words'][word]['count']+=1
                count = training[target]['words'][word]['count']
                training[target]['words'][word]['prob']= (count + 1.0) / (n + m)
        

        
    pp.pprint(training)
    print vocabulary 
    
def predict(X):
    positions = list(set(vocabulary) & set(X)) #remove words that are out of vocab.

    probabilities = {}
    
    for target in training.keys():
        prob_target = training[target]['prob']
        print "Target "+str(target)+" prob: "+str(prob_target)
        product = 1.0
        for word in positions:
            word_prob =  training[target]['words'][word]['prob']
            print "P("+str(word)+"|"+str(target)+") = "+str(word_prob)
            product *= word_prob
        target_prob = prob_target * product
        probabilities[target_prob]=target
        print "Target "+str(target)+" total prob. product: "+str(target_prob)
    print probabilities
    return  probabilities[max(probabilities)]

X=[[1,3,5],
    [1,4,9],
    [4,2,2]]
y=[1,2,11]

train(X,y)

Y=[1,4,9]
print predict(Y)

import numpy as np
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X, y)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
print("\n scikit: ")
print(clf.predict(Y))