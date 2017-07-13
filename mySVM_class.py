import numpy as np
#from sklearn.model_selection import KFold
import sklearn as sk
from smo import smo

def gaussian_kernel(x, y, sigma):
	return np.exp(-sigma*np.linalg.norm(x-y)**2)
	
def scalar_product(v1,v2):
    '''standard scalar product'''
    return np.dot(v1,v2)
	
def extract_suppvectors(alpha, data):
    supp_indices = np.array([i for i in range(len(data)) if not alpha[i]==0])
    supp_vectors = data[supp_indices]
    return supp_indices, supp_vectors

def extract_w(alpha, labels, data, supp_indices):
    #nur für linear kernels
    w = np.sum((np.array(alpha[i]*labels[i]*data[i]) for i in supp_indices), axis = 0)
    return w

def y_withoutb(x, alpha_supp, labels_supp, data_supp, kernel):
    alphatimeslabels = np.multiply(alpha_supp, labels_supp)
    k = np.array([kernel(x, y) for y in data_supp])
    return np.dot(alphatimeslabels, k)

def extract_b(alpha_supp, labels_supp, data_supp, kernel, C):
    indicesonmargin = np.array([i for i in range(len(data_supp)) if alpha_supp[i]<C ])
    return 1./len(indicesonmargin)*sum([labels_supp[i]-y_withoutb(data_supp[i],alpha_supp, labels_supp, data_supp, kernel) for i in indicesonmargin])
		

class mySVM:
    def __init__(self, kernel = gaussian_kernel, penalty = 1, sigma = 0.01):
        self.sigma = sigma
        self.kernel = kernel
        self.penalty = penalty
        self.kernelmatrix_for_predict = None
        self.tolerance = 1e-5
        
    def fit(self,training_data, training_labels):
        kernel_identifier = None
        if self.kernel == gaussian_kernel:
            def kernel_sigma(x, y):
                return gaussian_kernel(x, y, self.sigma)
            self.kernel = kernel_sigma
        if self.kernel == scalar_product:
            kernel_identifier = 'standard scalar product'
        solution = smo(np.transpose(training_data),training_labels,self.penalty,self.kernel,self.tolerance,'yes', kernel_identifier)
        self.alpha = solution['solution']
        self.training_data = training_data
        self.training_labels = training_labels
        self.supp_indices, self.supp_vectors = extract_suppvectors(self.alpha, training_data) #support vectors and indices of support vectors
        self.b = extract_b(self.alpha[self.supp_indices], training_labels[self.supp_indices], self.supp_vectors, self.kernel, self.penalty)
        if self.kernel == scalar_product:
            self.w = extract_w(self.alpha, training_labels, training_data, self.supp_indices)

    def decision_function(self, new_data):
        l = len(new_data)
        y = np.zeros(l)
        for i in range(l):
            x = new_data[i]
            k = np.array([self.kernel(y, x) for y in self.supp_vectors])
            atimeslabels_supp = np.multiply(self.alpha[self.supp_indices], self.training_labels[self.supp_indices])
            y[i] = np.dot(atimeslabels_supp, k) + self.b
        return y

def cross_validation(data,labels, penalty, kernel, sigma = .1):
    l = len(data)  
    kf = sk.model_selection.KFold(n_splits=2, shuffle=True)
    score = [0,0]
    i=0
    for train_index, test_index in kf.split(data):
        #split into training set and test set
        #training_data = data[train_index]
        #test_data = data[test_index]
        #training_labels = labels[train_index]
        #test_labels = labels[test_index]
        svm = mySVM(kernel=kernel, penalty=penalty, sigma=sigma)
        svm.fit(data[train_index], labels[train_index])
        predictions = np.sign(svm.decision_function(data[test_index]))
        score[i] = sum(predictions==labels[test_index])/float(len(test_index))
        i += 1
    score = 0.5*(score[0]+score[1])
    return score
