import loader as loader
import gaussian_classifier as clf
import pprint 
from sklearn import datasets
from sklearn.cross_validation import train_test_split
pp = pprint.PrettyPrinter(depth=6)

# digits = datasets.load_digits()
# X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.4, random_state=4)
# X_train.shape

X_train, Y_train = loader.loadmnist('gaussian_classifier/data/train-images-idx3-ubyte', 
'gaussian_classifier/data/train-labels-idx1-ubyte')
X_test, Y_test = loader.loadmnist('gaussian_classifier/data/t10k-images-idx3-ubyte', 
'gaussian_classifier/data/t10k-labels-idx1-ubyte')
print X_train.shape
print Y_train.shape

clf =  clf.GaussianClassifier(c=1)
print "training..."
clf.fit(X_train, Y_train)
print "classifying..."
Y = clf.predict(X_test)

errors = (Y_test != Y).sum()
total = X_test.shape[0]
print("Success rate:\t %d/%d = %f" % ((total-errors,total,((total-errors)/float(total)))))
print("Error rate:\t %d/%d = %f" % ((errors,total,(errors/float(total)))))



# 1     8412/10000 = 0.841200 1m25.659s
# 3000  9565/10000 = 0.956500 1m24.285s
# 3461  9565/10000 = 0.956500 1m23.864s
# 6461  9543/10000 = 0.954300 1m24.075s