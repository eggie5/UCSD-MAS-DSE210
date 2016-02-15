import loader as loader
import gaussian_classifier as gc
import pprint 
from sklearn import datasets, grid_search
from sklearn.cross_validation import train_test_split
pp = pprint.PrettyPrinter(depth=6)
from sklearn.metrics import classification_report

# digits = datasets.load_digits()
# X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.4, random_state=4)
# X_train.shape

X_train, Y_train = loader.loadmnist('gaussian_classifier/data/train-images-idx3-ubyte', 
'gaussian_classifier/data/train-labels-idx1-ubyte')
X_test, Y_test = loader.loadmnist('gaussian_classifier/data/t10k-images-idx3-ubyte', 
'gaussian_classifier/data/t10k-labels-idx1-ubyte')

print X_train.shape

# s=100
# X_train  = X_train[0:s]
# Y_train = Y_train[0:s]
# X_test = X_test[0:s]
# Y_test = Y_test[0:s]

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.5, random_state=4)
print X_train.shape
print Y_train.shape

_c=[1,10,100,1000,10000,100000]

for c in _c:
    clf =  gc.GaussianClassifier(c=c)
    print "training..."
    clf.fit(X_train, Y_train)
    print "classifying..."
    Y = clf.predict(X_test)

    errors = (Y_test != Y).sum()
    total = X_test.shape[0]
    print("Success rate:\t %d/%d = %f" % ((total-errors,total,((total-errors)/float(total)))))
    print("Error rate:\t %d/%d = %f" % ((errors,total,(errors/float(total)))))
    print ""


# # Split the dataset in two equal parts
# X_train, X_test, y_train, y_test = train_test_split(
#     X_train, Y_train, test_size=0.5, random_state=0)
#
# print X_train.shape
#
# parameters = [
#   {'c': [1], 'cov_algo': ['numpy', 'EmpiricalCovariance']}
#  ]
#
# clf = grid_search.GridSearchCV(gc, parameters, scoring='accuracy', n_jobs=1)
# clf.fit(X_train, y_train)
#
# print("Best parameters set found on development set:")
# print()
# print(clf.best_params_)
# print()
# print("Grid scores on development set:")
# print()
# for params, mean_score, scores in clf.grid_scores_:
#     print("%0.3f (+/-%0.03f) for %r"
#           % (mean_score, scores.std() * 2, params))
# print()
#
# print("Detailed classification report:")
# print()
# print("The model is trained on the full development set.")
# print("The scores are computed on the full evaluation set.")
# print()
# y_true, y_pred = y_test, clf.predict(X_test)
# print(classification_report(y_true, y_pred))
# print()