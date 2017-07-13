def m_classes(l,n,m):
    #l: samples per class
    #n: dimension
    #m: number of classes
    
    centers = 20*np.random.rand(m,n)-10 #create the centers of the classes
    data = np.zeros((1, n))
    labels = np.array([])
    for i in range( m):
        center = centers[i]
        print(np.shape(data), np.shape(np.random.normal(loc=center, scale=1.0, size=(l, n))))
        
        data = np.concatenate((data,np.random.normal(loc=center, scale=1.0, size=(l, n))), axis = 0)
        
        label = np.concatenate((labels, ))
    return data[1:], labels, centers
