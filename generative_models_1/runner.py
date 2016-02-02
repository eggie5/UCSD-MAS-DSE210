import naive_bayes_classifier as lib

clf =  lib.NaiveBayesClassifier("hi")
# X=[[1,2],[3,4],[5,6]]
# y=[1, 2, 3]
# clf.train(X,y)
# category = clf.predict([1,2,3,4,5,6])
# print category

# import impl as nb
import utils as utils
import numpy as np


path = '20news-bydate-matlab/matlab'
features = utils.read_features("expanded.txt")
label_array = utils.read_label(path, 'train.label')
print len(label_array)




answer_label_array = utils.read_label(path, 'test.label')
test_features = utils.read_features("test_expanded.txt")

vocab = utils.read_vocab("vocabulary.txt")

example_count=11269# total examples in training set
#nb.train(features[0:example_count,],label_array[0:example_count,])
clf.train(features[0:example_count], label_array[0:example_count], vocab=vocab)

#
# Y=[9,6]
# print "\nTrying to predict: "+str(Y)

correct_count = 0
for i in range(len(answer_label_array[0:example_count])):
    # pred_answer= nb.predict(test_features[i])
    pred_answer = clf.predict(test_features[i])
    actual_answer = answer_label_array[i]
    if pred_answer == actual_answer:
        # print "."
        correct_count+=1
    else:
        pass
        # print "expected {0}, actual: {1}".format(pred_answer, actual_answer)
        

print "{0}/{1}".format(correct_count, len(answer_label_array[0:example_count]))
print float(correct_count)/len(answer_label_array[0:example_count])





# X=[[1,3,5],
#     [1,4,9],
#     [4,2,2],
#     [9,6,9,6]]
# y=[1,2,1,3]
#
# train(X,y)
#
# Y=[9,6]
# print "\nTrying to predict: "+str(Y)
# print predict(Y)

# import numpy as np
# from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB()
# clf.fit(X, y)
# MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
# print("\n scikit: ")
# print(clf.predict(Y))