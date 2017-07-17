import numpy as np
import sklearn as sk
from smo_wss2_new_fettesK import smo_new
import scipy
import functools
import pickle
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.optimize import minimize
from scipy.spatial.distance import hamming


def gaussian_kernel(x, y, sigma=.1):
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
        self.tolerance = 1e-2
        
    def fit(self,training_data, training_labels):
        kernel_identifier = None
        if self.kernel == gaussian_kernel:
            self.kernel = functools.partial(gaussian_kernel, sigma = self.sigma)
        if self.kernel == scalar_product:
            kernel_identifier = 'standard scalar product'
        solution = smo_new(training_data,training_labels,self.penalty,self.kernel,self.tolerance,'yes',kernel_identifier)
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


def cross_validation_ecoc(data, labels, penalty, kernel=scalar_product, sigma = .1):
    l = len(data)  
    kf = sk.model_selection.KFold(n_splits=2, shuffle=True)
    score = [0,0]
    i=0
    for train_index, test_index in kf.split(data):

        ecoc_labels, list_supp_ind, list_alpha, list_b, list_kernel, code_words = ecoc(
            data[train_index],
            labels[train_index],
            penalty = penalty
        )
        final_labels = predict_ecoc(
            data[test_index],
            data[train_index],
            ecoc_labels,
            list_supp_ind,
            list_alpha,
            list_b,
            list_kernel,
            code_words
        )

        score[i] = sum(final_labels==labels[test_index])/float(len(test_index))
        i += 1
    score = 0.5*(score[0]+score[1])
    return score

def ecoc(labeled_data, labels, kernel=scalar_product, penalty_list=[10]*15, list_sigma=[0.1]*15):
    # 
    labels=labels.astype(int)
    l=np.shape(labeled_data)[0]
    num_classifiers=15
    ecoc_labels=np.zeros((l,15))
    
    # compute barycenters of the points of each label
    barycenters = np.zeros((10,np.shape(labeled_data)[1]));
    
    for i in range(10):
        #ind = labels == i
        ind = [j for j,k in enumerate(labels) if k == i]
        barycenters[i] = np.mean(labeled_data[ind], axis=0)
    
    
    # define code_word matrix, the ith row corresponds to the number i
    # each column corresponds to a classifier that will have to be trained
    code_words=np.array([
        [ 1,  1, -1, -1, -1, -1,  1, -1,  1, -1, -1,  1,  1, -1,  1],
        [-1, -1,  1,  1,  1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1],
        [ 1, -1, -1,  1, -1, -1, -1,  1,  1,  1,  1, -1,  1, -1,  1],
        [-1, -1,  1,  1, -1,  1,  1,  1, -1, -1, -1, -1,  1, -1,  1],
        [ 1,  1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1],
        [-1,  1, -1, -1,  1,  1, -1,  1,  1,  1, -1, -1, -1, -1,  1],
        [ 1, -1,  1,  1,  1, -1, -1, -1, -1,  1, -1,  1, -1, -1,  1],
        [-1, -1, -1,  1,  1,  1,  1, -1,  1, -1,  1,  1, -1, -1,  1],
        [ 1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1],
        [-1,  1,  1,  1, -1, -1, -1, -1,  1, -1,  1, -1, -1,  1,  1]])
    
    # up until now training data has labels from 0 to 9
    # now these are replaced by the 15 digit string given by code_words
    for j in range(l):
        ecoc_labels[j]=code_words[labels[j]]
    
    list_supp_ind = []
    list_alpha =[]
    list_b =[]
    list_kernel=[]
    
    # class an svm object for each classifier
    # here would be the possibility to parallelize
    for classifier in range(15):
        svm=mySVM(kernel=kernel, penalty=penalty_list[classifier], sigma=list_sigma[classifier])
        svm.fit(labeled_data, ecoc_labels[:,classifier])
        list_supp_ind.append(svm.supp_indices)
        list_alpha.append(svm.alpha)
        list_b.append(svm.b)
        list_kernel.append(svm.kernel)

    # pickle dump to save and call saved objects    
    pickle.dump((ecoc_labels, list_supp_ind, list_alpha, list_b, list_kernel, code_words, barycenters), open( "trained_ecoc_"+str(l)+".dat", "wb" ))
    
    # now I need to call a binary classifier for each column of ecoc_labels
    # from decision functions we get seperating hyperplanes, margin, ... 
    # return those
    
    
    return ecoc_labels, list_supp_ind, list_alpha, list_b, list_kernel, code_words, barycenters



# suppose we have an unlabeled data point
def predict_ecoc(unlabeled_data, labeled_data, ecoc_labels, list_supp_ind, list_alpha, list_b, list_kernel, code_words, barycenters):
    # every row is one data point
    # number of rows = # of data points
    l=np.shape(unlabeled_data)[0]
    new_labels=np.zeros((l,15))
    
    temp_label_ind=[]
    final_labels = np.array([float('inf')]*l)
    
    for classifier in range(15):
        a_supp = list_alpha[classifier][list_supp_ind[classifier]]
        ecoc_labels_supp = ecoc_labels[list_supp_ind[classifier],classifier]
        a_times_labels=np.multiply(a_supp, ecoc_labels_supp)
        
        for i in range(l):
            # i_th row of kernel matrix k
            k=np.array([list_kernel[classifier](unlabeled_data[i],y) for y in labeled_data[list_supp_ind[classifier]]])
            
            # list of lists with 15 entries, one per classifier
            new_labels[i][classifier]=np.sign(np.dot(a_times_labels,k)+list_b[classifier])
    
        
    for i in range(l):
        ham_dist = [hamming(new_labels[i], code_words[j]) for j in range(10)]
        temp_label_ind = [j for j in range(len(ham_dist)) if ham_dist[j] == min(ham_dist)]
        if len(temp_label_ind)!=1:
            print("Attention, data point could not be uniquely classified, index: " 
                  + str(i) + ", possible classification: " + str(temp_label_ind))
            # ask which barycenter is closest out of temp_label_ind
            final_labels[i] = temp_label_ind[np.argmin([np.linalg.norm(unlabeled_data[i]-barycenters[k]) for k in temp_label_ind])]
            print("decided to take label: ", final_labels[i])
        else:
            final_labels[i] = ham_dist.index(min(ham_dist))
        
   
    return final_labels.astype(float)
    
