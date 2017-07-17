import numpy as np

# careful, smo expects data in the shape of a dxl array, where d is the dimension of the data points and l the number of data points!
def smo(data,label,C,kernel,tol,violationcheckyesorno, kernel_identifier = None):
    
    eps = 1e-12
    
    l = label.shape[0]
    alpha = np.zeros(l)
    
    fcache = -label.astype(float)
    
    b_up = [-1]
    b_low = [1]
    
    I = np.zeros((5,l), dtype=np.int)
    I[1,:] = (label == 1).astype(int)
    I[4,:] = (label == -1).astype(int)

    i_up = [min(np.nonzero(I[1,:])[0])]
    i_low = [min(np.nonzero(I[4,:])[0])]
    
    
    ##############################################################################################
    # auxiliary functions for takeStep and examineExample
    
    def I_membership(a,y):
        if 0 < a and a < C:
            return np.array([1,0,0,0,0])
        elif a == 0 and y == 1:
            return np.array([0,1,0,0,0])
        elif a == C and y == -1:
            return np.array([0,0,1,0,0])
        elif a == C and y == 1:
            return np.array([0,0,0,1,0])
        else:
            return np.array([0,0,0,0,1])
    

    def I_membership_no(a,y):
        if 0 < a < C:
            return 0
        elif a == 0 and y == 1:
            return 1
        elif a == C and y == -1:
            return 2
        elif a == C and y == 1:
            return 3
        else:
            return 4
        
    def F(i,alpha):
        
        #avoid for loop in case of standard scalar product kernel
        if kernel_identifier == 'standard scalar product':
            k = alpha * label;
            m = np.dot(data[:,i],data)
            out = -label[i] + np.dot(k,m)
        else: 
            out = - label[i]
            for j in range(l):
                out += alpha[j]*label[j]*kernel(data[:,i],data[:,j])
        return out
    
    
    ##############################################################################################   
    
    
    
    
    ##############################################################################################
    # subprocedure takeStep
    
    def takeStep(i1,i2):

    
        if i1 == i2: 
            return 0
        alph1 = alpha[i1]
        alph2 = alpha[i2]
        y1 = label[i1]
        y2 = label[i2]  
   
        F1 = fcache[i1]
        F2 = fcache[i2]
        s = y1*y2
    
        if s == -1:
            L = max(0,alph2-alph1)
            H = min(C,C+alph2-alph1)
        else:
            L = max(0,alph2+alph1-C)
            H = min(C,alph2+alph1)
    
        if L == H: return 0
    
        k11 = kernel(data[:,i1],data[:,i1])
        k12 = kernel(data[:,i1],data[:,i2])
        k22 = kernel(data[:,i2],data[:,i2])
    
        eta = 2*k12 - k11 - k22
        if eta < 0:
            a2 = alph2 - y2*(F1-F2)/eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            #print('Error: eta == 0')
            raise ValueError('Error: eta == 0')
    
        if abs(a2-alph2) < eps*(a2+alph2+eps):
            return 0
    
        a1 = alph1 + s*(alph2-a2)
    
   
        # update alpha
        alpha[i1] = a1
        alpha[i2] = a2
    
        # update I
        I[:,i1] =  I_membership(a1,y1)
        I[:,i2] =  I_membership(a2,y2)
    
        # update fcache[i] here for i in I_0: a[0:3][:,4:9]
        ind_0 = (np.nonzero(I[0,:]))[0]
        
        #avoid for loop in case of standard scalar product kernel
        if kernel_identifier == 'standard scalar product':
            fcache[ind_0] += y1*(a1-alph1)*np.dot(data[:,i1],data[:,ind_0]) + y2*(a2-alph2)*np.dot(data[:,i2],data[:,ind_0])
        else:
            for i in ind_0:
                fcache[i] = fcache[i] + y1*(a1-alph1)*kernel(data[:,i],data[:,i1]) + y2*(a2-alph2)*kernel(data[:,i],data[:,i2])
    
        # update fcache for indices i1 and i2
        fcache[i1] = F1 + y1*(a1-alph1)*k11 + y2*(a2-alph2)*k12
        fcache[i2] = F2 + y1*(a1-alph1)*k12 + y2*(a2-alph2)*k22
        
        
        b_low[0] = -float('inf')
        b_up[0] = - b_low[0]
        
        if np.size(ind_0) > 0:
            for i in ind_0:
                if fcache[i] > b_low[0]:
                    b_low[0] = fcache[i]
                    i_low[0] = i
                if fcache[i] < b_up:
                    b_up[0] = fcache[i]
                    i_up[0] = i
       
        memb1 = I_membership_no(a1,y1)
        memb2 = I_membership_no(a2,y2)
        if memb1 in (1,2) and fcache[i1] < b_up[0]:
            b_up[0] = fcache[i1]
            i_up[0] = i1
        if memb1 in (3,4) and fcache[i1] > b_low[0]:
            b_low[0] = fcache[i1]
            i_low[0] = i1
        if memb2 in (1,2) and fcache[i2] < b_up[0]:
            b_up[0] = fcache[i2]
            i_up[0] = i2
        if memb2 in (3,4) and fcache[i2] > b_low[0]:
            b_low[0] = fcache[i2]
            i_low[0] = i2

        return 1
    
    ##############################################################################################
    
    
    
    
    ##############################################################################################
    #subprocedure examineExample
    
    def examineExample(i2):

    
        y2 = label[i2]
        alph2 = alpha[i2]
        memb_i2 = I_membership_no(alph2,y2);
    
        # look up or compute F2:= F_i2
        if memb_i2 == 0:
            F2 = fcache[i2]
        else:
            fcache[i2] = F(i2,alpha)
            F2 = fcache[i2]
        
            # in case i2 not in I_0, see if F2 changes or estimate of b_low or b_up
            if memb_i2 in (1,2) and F2 < b_up[0]:
                b_up[0] = F2
                i_up[0] = i2
            elif memb_i2 in (3,4) and F2 > b_low[0]:
                b_low[0] = F2
                i_low[0] = i2
    
        optimality = 1
        if memb_i2 in (0,1,2):
            if b_low[0] - F2 > tol:
                optimality = 0
                i1 = i_low[0]
        if memb_i2 in (0,3,4):
            if F2 - b_up[0] > tol:
                optimality = 0
                i1 = i_up[0]
        if optimality == 1:
            return 0
    
        if memb_i2 == 0:
            if b_low[0] - F2 > F2 - b_up[0]:
                i1 = i_low[0]
            else:
                i1 = i_up[0]
            
        if takeStep(i1,i2):
            return 1
        else:
            return 0
    
    ##############################################################################################
    
    
    
    
    ##############################################################################################
    #main routine
    
    numChanged = 0
    examineAll = 1

    while numChanged > 0 or examineAll:
        numChanged = 0
        if examineAll:
            for i in range(l):
                numChanged = numChanged + examineExample(i)
        else:
            for i in (np.nonzero(I[0,:]))[0]:
                numChanged = numChanged + examineExample(i)
                if b_up[0] > b_low[0] - tol:
                    numChanged = 0
                    break
        if examineAll == 1:
            examineAll = 0
        elif numChanged == 0:
            examineAll = 1
            
            
    if violationcheckyesorno == 'yes':
        memb = [I_membership_no(alpha[i],label[i]) for i in range(l)]
        I_up = [memb[i] in (0,1,2) for i in range(l)]
        I_low = [memb[i] in (0,3,4) for i in range(l)]
        bs = [F(i,alpha) for i in range(l)]

        b_up = float('inf')
        b_low = -b_up
        for i in range(l):
            if I_up[i] == 1 and bs[i] < b_up:
                b_up = bs[i]
            if I_low[i] == 1 and bs[i] > b_low:
                b_low = bs[i]

        if b_low - tol <= b_up:
            violationstring = 'no tol-violation'
        else:
            violationstring = 'tol-violations!!'
    else:
        violationstring = 'no violation requested'

    
    return {'solution': alpha, 'violationcheck': violationstring}

    ##############################################################################################
