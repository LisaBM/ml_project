import numpy as np
import scipy

# to save data
import pickle
from scipy.optimize import minimize
from scipy.spatial.distance import hamming

from mySVM_class import *


def oneVSall(labeled_data, labels, num_classifiers=10, kernel=scalar_product, penalty=10, list_sigma=[0.003]*10):
    # 
    labels=labels.astype(int);
    l=np.shape(labeled_data)[0];
    n=np.shape(labeled_data)[1];
    #num_classifiers=3;
    oneVSall_labels=np.zeros((l,num_classifiers));
    
    # define code_word matrix, the ith row corresponds to the number i
    # each column corresponds to a classifier that will have to be trained
    code_words=2*np.identity(num_classifiers)-1;
    
    # up until now training data has labels from 0 to 9
    # now these are replaced by the 15 digit string given by code_words
    for j in range(l):
        oneVSall_labels[j]=code_words[labels[j]];
    
    list_supp_ind = [];
    list_alpha =[];
    list_b =[];
    list_kernel=[];
    list_w=[];
    
    # print(oneVSall_labels)
    
    # class an svm object for each classifier
    # here would be the possibility to parallelize
    for classifier in range(num_classifiers):
        svm=mySVM(kernel=kernel, penalty=penalty, sigma=list_sigma[classifier]);
        svm.fit(labeled_data, oneVSall_labels[:,classifier]);
        list_supp_ind.append(svm.supp_indices);
        list_alpha.append(svm.alpha);
        list_b.append(svm.b);
        list_kernel.append(svm.kernel);
        # list_w.append(svm.w)
        
    # compute barycenters of the points of each label
    barycenters = np.zeros((num_classifiers,n));
    # barycenters2 = np.zeros((num_classifiers,n));
    for i in range(num_classifiers):
        # calculations of the barycenters yield same results
        # so probably both are correct... 
        ind = [j for j,k in enumerate(labels) if k == i]
        barycenters[i] = np.mean(labeled_data[ind], axis=0)
        # ind2 = labels == i
        # barycenters2[i] = np.mean(labeled_data[ind], axis=0)
        # print(np.linalg.norm(barycenters - barycenters2))
    # pickle dump to save and call saved objects    
    # pickle.dump((oneVSall_labels, list_supp_ind, list_alpha, list_b, list_kernel, code_words, barycenters), open( "trained_oneVSall_"+str(number_of_im)+".dat", "wb" ))
    
    # now I need to call a binary classifier for each column of ecoc_labels
    # from decision functions we get seperating hyperplanes, margin, ... 
    # return those
    return oneVSall_labels, list_supp_ind, list_alpha, list_b, list_kernel, code_words, barycenters;
    
    

def predict_oneVSall(unlabeled_data, labeled_data, oneVSall_labels, list_supp_ind, 
                     list_alpha, list_b, list_kernel, code_words, barycenters, num_classifiers=10):
    # every row is one data point
    # number of rows = # of data points
    #num_classifiers=3;
    
    l=np.shape(unlabeled_data)[0];
    new_labels=np.zeros((l,num_classifiers));
    
    temp_label_ind=[];
    final_labels = np.array(['inf']*l).astype(float);
    
    counter_onevsall = 0;
    counter_barycenter=0;
    list_oneVsall_unique_index=[];
    
    for classifier in range(num_classifiers):
        a_supp = list_alpha[classifier][list_supp_ind[classifier]];
        oneVSall_labels_supp = oneVSall_labels[list_supp_ind[classifier],classifier]
        a_times_labels=np.multiply(a_supp, oneVSall_labels_supp)
        
        for i in range(l):
            # i_th row of kernel matrix k
            k=np.array([list_kernel[classifier](unlabeled_data[i],y) for y in labeled_data[list_supp_ind[classifier]]])
            
            # list of lists with 15 entries, one per classifier
            new_labels[i][classifier]=np.sign(np.dot(a_times_labels,k)+list_b[classifier]);
    
    for i in range(l):
        ham_dist = [hamming(new_labels[i], code_words[j]) for j in range(num_classifiers)]
        temp_label_ind = [j for j in range(len(ham_dist)) if ham_dist[j] == min(ham_dist)]
        # print(type(temp_label_ind[0]))
        if len(temp_label_ind)!=1:
            # print("Attention, data point could not be uniquely classified, index " 
              #    + str(i) + " possible classification " + str(temp_label_ind));
            
            # ask which barycenter is closest out of temp_label_ind
            final_labels[i] = np.min(np.argmin([np.linalg.norm(unlabeled_data[i]-barycenters[k]) for k in temp_label_ind]))
            # final_labels[i]=20;
            counter_barycenter+=1;
        else:
            counter_onevsall += 1;
            final_labels[i] = ham_dist.index(min(ham_dist));
            list_oneVsall_unique_index.append(i); 
            
        
    print("counter one Vs all classified ", counter_onevsall)
    print("counter barycenter classified ", counter_barycenter)
    return final_labels, list_oneVsall_unique_index;
    
    
