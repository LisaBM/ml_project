def gaussian_kernel(x, y, sigma):
	return np.exp(-sigma*np.linalg.norm(x-y)**2)

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
            self.w = extract_w(self.alpha, training_labels, training_data)
			
def extract_suppvectors(alpha, data):
    supp_indices = np.array([i for i in range(len(data)) if not alpha[i]==0])
    supp_vectors = data[supp_indices]
    return supp_indices, supp_vectors

def extract_w(alpa, labels, data):
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
	
def scalar_product(v1,v2):
    '''standard scalar product'''
    return np.dot(v1,v2)
	
