import numpy as np
import sklearn as sk
from smo_wss1 import smo
import scipy
import functools
import pickle
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.optimize import minimize
from scipy.spatial.distance import hamming


########################################
###        AUXILIARY FUNCTIONS       ###
########################################


# We call the SVM with two different kernel functions:

def gaussian_kernel(x, y, sigma=.1):
    return np.exp(-sigma*np.linalg.norm(x-y)**2)

def scalar_product(v1,v2):
    '''standard scalar product'''
    return np.dot(v1,v2)


# From the data that our SMO returns, we want to reconstruct the support vectors, their indices, the vector w describing the hyperplane (in case of the linear kernel), the decision function including the offset b.

def extract_suppvectors(alpha, data):
    supp_indices = np.array([i for i in range(len(data)) if not alpha[i]==0])
    supp_vectors = data[supp_indices]
    return supp_indices, supp_vectors

def extract_w(alpha, labels, data, supp_indices):
    #nur für linear kernels
    w = np.sum((np.array(alpha[i]*labels[i]*data[i]) for i in supp_indices), axis = 0)
    return w

def extract_b(alpha_supp, labels_supp, data_supp, kernel, C):
    indicesonmargin = np.array([i for i in range(len(data_supp)) if alpha_supp[i]<C ])
    return 1./len(indicesonmargin)*sum([labels_supp[i]-y_withoutb(data_supp[i],alpha_supp, labels_supp, data_supp, kernel) for i in indicesonmargin])

def y_withoutb(x, alpha_supp, labels_supp, data_supp, kernel):
    alphatimeslabels = np.multiply(alpha_supp, labels_supp)
    k = np.array([kernel(x, y) for y in data_supp])
    return np.dot(alphatimeslabels, k)


########################################
###           OUR SVM CLASS          ###
########################################

class mySVM:
    
    # Create an instance of our SVM by chosing a kernel and setting the parameters 
    def __init__(
        self,
        kernel = gaussian_kernel,
        penalty = 1,
        sigma = 0.01
    ):
        self.sigma = sigma
        self.kernel = kernel
        self.penalty = penalty
        self.kernelmatrix_for_predict = None
        self.tolerance = 1e-2
        
    # Fit the SVM to your set of trainings data by calling the fit function. 
    def fit(
        self,
        training_data,
        training_labels
    ):
        kernel_identifier = None
        if self.kernel == gaussian_kernel:
            self.kernel = functools.partial(gaussian_kernel, sigma = self.sigma)
        if self.kernel == scalar_product:
            kernel_identifier = 'standard scalar product'
        
        # Store the solution alpha of the dual optimization problem which is specified by the attributes of the instance of mySVM as well as the trainings data and trainings labels
        alpha = smo(
            training_data,
            training_labels,
            self.penalty,
            self.kernel,
            self.tolerance,
            'no', #put yes if you want to know whether there were any violations of the tolerance
            kernel_identifier #need this for the technical reasons
        )
        self.alpha = alpha['solution']
        self.training_data = training_data
        self.training_labels = training_labels
        
        ### Extract all further information by using the auxiliary functions defined above: ###
        #support vectors and indices of support
        self.supp_indices, self.supp_vectors = extract_suppvectors(self.alpha, training_data)
        self.b = extract_b(self.alpha[self.supp_indices], training_labels[self.supp_indices], self.supp_vectors, self.kernel, self.penalty)
        if self.kernel == scalar_product:
            self.w = extract_w(self.alpha, training_labels, training_data, self.supp_indices)

    #Given new data points, use the above extracted variables to define the decision function that gives the decision boundary at level 0. We did not implement the prediction function itself since it is just the sgn of this decision function.
    def decision_function(self, new_data):
        l = len(new_data)
        y_vector = np.zeros(l) #initializing the vector that will store the value for each new data point
        atimeslabels_supp = np.multiply(self.alpha[self.supp_indices], self.training_labels[self.supp_indices]) #vector consisting of entries alpha_i*label_i
        
        #for each of the new data points, calculate the respective column in the kernel matrix and compute the value of the decision function
        for i in range(l):
            x = new_data[i]
            k = np.array([self.kernel(y, x) for y in self.supp_vectors])
            y_vector[i] = np.dot(atimeslabels_supp, k) + self.b
        return y_vector #These are not the labels yet, but sgn of this are the predicted labels


########################################
###         OUR ECOC MACHINES        ###
########################################


# The ECOC Machine trains 15 different SVMs on different binary classification of the data and then combines the results into a final prediction.
# The ecoc function is called with the labeled data, the kernel and the parameters which can be different for each of the 15 classifiers.

def ecoc(labeled_data, labels, kernel=scalar_product, penalty_list=[10]*15, list_sigma=[0.1]*15):
    labels = labels.astype(int)
    l = np.shape(labeled_data)[0]
    #Initializing the vector that will store the predicted labels:
    ecoc_labels = np.zeros((l,15))
    
    # Compute barycenters of all the data points belonging to the same label (out of the original 10 labels):
    barycenters = np.zeros((10, np.shape(labeled_data)[1]))
    for i in range(10):
        ind = [j for j,k in enumerate(labels) if k == i]
        barycenters[i] = np.mean(labeled_data[ind], axis=0)
    
    
    # Define the code word matrix. The i-th row represents the ecoc label of the number i
    # Each of the 15 columns corresponds to a classifier that will be trained according to a specific relabelling of the data (into two classes)
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
    
    # Up until now training data has labels from 0 to 9, now these are replaced by the 15 digit string given by code_words:
    for j in range(l):
        ecoc_labels[j]=code_words[labels[j]]
    
    #Initializing of several variables:
    list_supp_ind = []
    list_alpha =[]
    list_b =[]
    list_kernel=[]
    
    # Now we create an SVM object for each classifier. We train the i-th SVM using the labels given by the i-th column of code_words. We then store all the information that is later needed for the prediction.
    for classifier in range(15):
        svm=mySVM(kernel=kernel, penalty=penalty_list[classifier], sigma=list_sigma[classifier])
        svm.fit(labeled_data, ecoc_labels[:,classifier])
        list_supp_ind.append(svm.supp_indices)
        list_alpha.append(svm.alpha)
        list_b.append(svm.b)
        list_kernel.append(svm.kernel)

    # We pickle dump (i.e. save) the collected information in a .dat-file in order to access it at any time. 
    pickle.dump(
        (ecoc_labels, list_supp_ind, list_alpha, list_b, list_kernel, code_words, barycenters),
        open( "trained_ecoc_"+str(l)+".dat", "wb" )
    )
    
    return ecoc_labels, list_supp_ind, list_alpha, list_b, list_kernel, code_words, barycenters


# The predict_ecoc function is called with the unlabeled data, the labeled data that was used for training the ECOC machine and also all the information that was returned by the ecoc function itself.
def predict_ecoc(
    unlabeled_data,
    labeled_data,
    ecoc_labels,
    list_supp_ind,
    list_alpha,
    list_b,
    list_kernel,
    code_words,
    barycenters
):
    # every row is one data point
    # number of rows = # of data points
    l = np.shape(unlabeled_data)[0]
    new_labels = np.zeros((l,15))
    
    temp_label_ind = []
    final_labels = np.array([float('inf')]*l)
    
    #For each of the 15 classifiers, predict the labels (i.e. +1/-1) with respect to the relabelling in the respective column of code_words
    for classifier in range(15):
        # Extract the vector a_supp consisting of all non-zero entries of alpha
        # and ecoc_labels_supp consisting of all ecoc_labels referring to support vectors.
        a_supp = list_alpha[classifier][list_supp_ind[classifier]]
        ecoc_labels_supp = ecoc_labels[list_supp_ind[classifier], classifier]
        a_times_labels = np.multiply(a_supp, ecoc_labels_supp)
        
        for i in range(l):
            # i-th row of kernel matrix
            k = np.array([
                list_kernel[classifier](unlabeled_data[i],y)
                for y in labeled_data[list_supp_ind[classifier]]
            ])
            # We then store for each new data point the new ecoc-label which is given by a list with 15 entries (one per classifier)
            # Note that these ecoc-labels must not appear as a row in code_words!
            new_labels[i][classifier]=np.sign(np.dot(a_times_labels,k)+list_b[classifier])
    
    # Now, for each new data point, convert the ecoc-label into a label from 0 to 9 by checking which code word is closest (wrt. the Hamming distance) to it. We then chosoe the digit corresponding to this code word.
    for i in range(l):
        # Compute the Hamming distance of the ecoc-label of the i-th data point to the 10 different code words and store the indices of the code words that have minimal distance.
        ham_dist = [hamming(new_labels[i], code_words[j]) for j in range(10)]
        temp_label_ind = [j for j in range(len(ham_dist)) if ham_dist[j] == min(ham_dist)]
        # If there is not a unique code word that minimizes the Hamming distance, choose from the left-over possibilites by minimizing the Euclidean distance to the barycenters of the 10 classes. (cf. Barycenters above)
        if len(temp_label_ind)!=1:
            print("Attention, data point could not be uniquely classified, index: " 
                  + str(i) + ", possible classification: " + str(temp_label_ind))
            # ask which barycenter is closest out of temp_label_ind
            final_labels[i] = temp_label_ind[np.argmin([np.linalg.norm(unlabeled_data[i]-barycenters[k]) for k in temp_label_ind])]
            print("decided to take label: ", final_labels[i])
        else:
            final_labels[i] = ham_dist.index(min(ham_dist))
        
    # Return the list that stores all the predicted labels as integers from 0 to 9
    return final_labels.astype(float)
    




########################################
###          CROSSVALIDATION         ###
########################################

# We define a function in order to cross-validate our algorithms: first the pure SVM with two classes and then the multi-class algorithm using ECOC

# These functions are called with the labelled data, the choice of kernel and parameters C, sigma. Then the data is randomly split into to parts of equal size and the algoritm is called on one side. Then we predict the other side and check what share of these was predicted correctly. We do this for both sides and average over the results. We thus return what we call the score.

def cross_validation(data, labels, penalty, kernel, sigma = .1):
    l = len(data)  
    kf = sk.model_selection.KFold(n_splits=2, shuffle=True) #Initialize a 2-fold separation
    score = [] #initializing the list that will store the scores for each side.
    for train_index, test_index in kf.split(data): # Our kf splits the data into training set and test set
        # We initialize and fit a mySVM on the train set and train labels
        svm = mySVM(kernel=kernel, penalty=penalty, sigma=sigma)
        svm.fit(data[train_index], labels[train_index])
        # We  predict the test set and compare the outcome to the actual labels
        predictions = np.sign(svm.decision_function(data[test_index]))
        score.append(sum(predictions==labels[test_index])/float(len(test_index)))
    # We average over both scores:
    score = 0.5*(score[0]+score[1])
    return score


def cross_validation_ecoc(data, labels, penalty, kernel=scalar_product, sigma = .1):
    l = len(data)  
    kf = sk.model_selection.KFold(n_splits=2, shuffle=True)
    score = []#initializing the list that will store the scores for each side.
    for train_index, test_index in kf.split(data): # Our kf splits the data into training set and test set
        # We initialize and fit an ecoc-machine on the train set and train labels
        ecoc_labels, list_supp_ind, list_alpha, list_b, list_kernel, code_words = ecoc(
            data[train_index],
            labels[train_index],
            penalty = penalty
        )
        # We  predict the test set and compare the outcome to the actual labels
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
        score.append(sum(final_labels==labels[test_index])/float(len(test_index)))
    # We average over both scores:
    score = 0.5*(score[0]+score[1])
    return score

