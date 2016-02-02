import numpy as np
from collections import Counter

class NaiveBayesClassifier(object):
    """docstring for NaiveBayesClassifier"""
    
    def __init__(self, arg):
        super(NaiveBayesClassifier, self).__init__()
        self.arg = arg
        self.training = {}
        self.vocabulary = set()
    
    def train(self, features, targets, vocab=None):
        target_count = len(set(targets))
        example_count = len(features)
        print("Total training examples: ")+str(example_count)
        print("Total targets: ")+str(target_count)
    
        #add words to vocabulary set
        if vocab is None:
            print "Building vocabulary from corpus"
            [self.vocabulary.add(word) for feature in features for word in feature]
        else:
            print "Buildign vocabulary from user input"
            self.vocabulary=vocab

        # lets get the vocab from vocabulary.txt
        vocab_count = len(self.vocabulary)
        print "Vocabulary count: "+str(vocab_count)
        print "Estimated iterations: "+str(vocab_count * target_count)
    
        #preprocessing -- need these constants before calculating posteri
        print "training step 1..."
        for i in range(example_count):
            print "Training step #1 for target: "+str(i)+" ..."
            example = features[i]
            target = targets[i]
        
            #setup the internal data structure
            if(target not in self.training):
                self.training[target]={'count':0, 'prob': 0.0, 'blob': [], 'n': 0, 'words':{}}
            
            #put the example doc words in the category bag
            self.training[target]['blob']+=example
            self.training[target]['n'] = len(list(set(self.training[target]['blob'])))

            #calc. priori probabilities
            self.training[target]['count']+=1
            self.training[target]['prob']= float(self.training[target]['count']) / float(len(targets))

    
        print "training step 2..."
        #now iterate the vocabulary for each category
        for i in range(target_count):
            print "Training step #2 for target: "+str(i)+" ..."
            
            target = targets[i]
        
            m=len(self.vocabulary)
            n = self.training[target]['n']
            norm = n+m
 
            for word in self.vocabulary:
                if(word not in self.training[target]['words']):
                    # in the vocab but not in the blob
                    self.training[target]['words'][word]={'count':0, 'prob':1.0/norm}

                if(word in self.training[target]['blob']):
                    self.training[target]['words'][word]['count']+=1
                    count = self.training[target]['words'][word]['count']
                    self.training[target]['words'][word]['prob']= (count + 1.0) / norm
        
        pp.pprint(self.training)
        return True

        

        # print vocabulary
    
    def predict(self, Y):
        print ("running predictions...")
        #TODO: Fix -> this removes duplicates, e.g.: Y=[1,2,3,9,9,9,9]
        positions = Y#list(set(self.vocabulary) & set(Y)) #remove words that are out of vocab.

        probabilities = {}

        for target in self.training.keys():
            prob_target = self.training[target]['prob']
            # print "Target "+str(target)+" prob: "+str(prob_target)
            # product = 1.0
            rsum=0
            for word in positions:
                word_prob =  self.training[target]['words'][word]['prob']
                # print "P("+str(word)+"|"+str(target)+") = "+str(word_prob)
                # product *= word_prob
                rsum+= np.log2(word_prob)
            # target_prob = prob_target * product
            target_prob = np.log2(prob_target) + rsum
            probabilities[target_prob]=target
            # print "Target "+str(target)+" total prob. product: "+str(target_prob)
            # print ""
        # print probabilities
        return  probabilities[max(probabilities)]
        
if __name__ == "__main__":
    import pprint 
    pp = pprint.PrettyPrinter(depth=6)
    
    clf =  NaiveBayesClassifier("hi")
    X=[[1,2],[3,4],[5,6], [1,2]]
    y=[1, 2, 3, 1]
    clf.train(X,y, vocab=[1,2,3,4,5,6,7,8,9])
    category = clf.predict([1,2])
    print category