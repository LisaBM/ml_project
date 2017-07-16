import multiprocessing as mp
import numpy as np
import pandas as pd
from mySVM_class_new_multiprocessing import *
import time




#random test problem


d = 10
l = 100
data1x = np.random.randint(255, size=(d-1,l))
data1y = np.random.randint(255, size=(1,l))
data2x = np.random.randint(255, size =(d-1,l))
data2y = -np.random.randint(-20,255, size=(1,l))
data1 = np.concatenate((data1x,data1y))
data2 = np.concatenate((data2x,data2y))
data = np.concatenate((data1,data2), axis = 1)
data = np.transpose(data)

label1 = np.ones(l).astype(int)
label2 = - label1;
label = np.concatenate((label1,label2))

l = 2*l


def kernel(v1,v2):
    """standard scalar product"""
    return np.dot(v1,v2)
kernel_identifier = kernel.__doc__


train = pd.read_csv('Data/train.csv', nrows = 10000)
images = ["%s%s" %("pixel",pixel_no) for pixel_no in range(0,28**2)]
train_images = np.array(train[images], dtype=np.float)/100
train_labels = np.array(train['label'])


no_train = 1000
train = train_images[:no_train]
train_l = train_labels[:no_train]
test = train_images[no_train:no_train+400]
test_l = train_labels[no_train:no_train+400]
lambda_opt = 1./(400*np.array([18,18,18,18,17,18,16,16,16,18,18,20,20,18,18])) #optimal lambdas found via cross validation
#C_list = 1./(2*len(train)*lambda_opt)  #compute optimal C from optimal lambda
#sigma_list = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.0025, 0.005, 0.005, 0.005, 0.0025, 0.005] #optimal sigmas found via cross validation
C_list = 1.
sigma_list = 0.005


start = time.time()


if __name__ == '__main__':
    output = mp.Queue()
    processes = [mp.Process(target=ecoc, args=(output, classifier, train, train_l, gaussian_kernel, C_list, sigma_list)) for classifier in range(4)]

    # Run processes
    for p in processes:
        p.start()

    results = [output.get() for p in processes]

    # Exit the completed processes
    for p in processes:
        p.join()

    # Get process results from the output queue

    print(results)

if __name__ == '__main__':
    output = mp.Queue()
    processes = [mp.Process(target=ecoc, args=(output, classifier, train, train_l,gaussian_kernel, C_list, sigma_list)) for classifier in range(4,8)]

    # Run processes
    for p in processes:
        p.start()

    results = [output.get() for p in processes]

    # Exit the completed processes
    for p in processes:
        p.join()

    # Get process results from the output queue

    print(results)


if __name__ == '__main__':
    output = mp.Queue()
    processes = [mp.Process(target=ecoc, args=(output, classifier, train, train_l,gaussian_kernel, C_list, sigma_list)) for classifier in range(8,12)]

    # Run processes
    for p in processes:
        p.start()

    results = [output.get() for p in processes]

    # Exit the completed processes
    for p in processes:
        p.join()

    # Get process results from the output queue

    print(results)


if __name__ == '__main__':
    output = mp.Queue()
    processes = [mp.Process(target=ecoc, args=(output, classifier, train, train_l,gaussian_kernel, C_list, sigma_list)) for classifier in range(12,15)]

    # Run processes
    for p in processes:
        p.start()

    results = [output.get() for p in processes]

    # Exit the completed processes
    for p in processes:
        p.join()

    # Get process results from the output queue

    print(results)

end = time.time()
print(end - start)