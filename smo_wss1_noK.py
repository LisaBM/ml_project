import numpy as np


def smo(data, label, C, kernel, tol, violationcheckyesorno,kernel_identifier = None):

    # returns logical vector of length 2 that states the membership of \alpha_i to I_up and I_low
    def I_up_low_membership(alpha_i, label_i):

        # critical parameter, if any termination problems occur, try increasing null a bit!
        null = 1e-15

        v = np.array([False, False])
        if (alpha_i < C - null and label_i == 1) or (alpha_i > null and label_i == -1):
            v[0] = True
        if (alpha_i < C - null and label_i == -1) or (alpha_i > null and label_i == 1):
            v[1] = True
        return v

    # data: images are rows of array, so the number of columns is 28**2 and the number of rows is the
    # number of training data

    # initialize all the central objects
    l = label.shape[0]
    alpha = np.zeros(l)
    I = np.zeros((2, l), dtype=bool)
    # I[0] = I_up, I[1] = I_low
    I[0] = label == 1
    I[1] = label == -1
    b_up = -1
    b_low = 1
    v = np.array(range(l))
    i_up_array = v[I[0]]
    i_low_array = v[I[1]]
    i_0 = i_up_array[0]
    j_0 = i_low_array[0]

    # cache the F values in order to apply an efficient update rule after each step
    fcache = -label.astype(float)

    # iteration counter, not needed for the functioning of this particular algorithm
    # iter = 0

    # two objects that will help determine if the algorithm got stuck, see the very bottom of the while loop
    stuckcache = -np.ones(2)
    stuckcounter = 0


    Ki0 = np.empty(l)
    Kj0 = np.empty(l)

    while (b_up < b_low - tol):

        alph1 = alpha[i_0]
        alph2 = alpha[j_0]
        y1 = label[i_0]
        y2 = label[j_0]
        F1 = fcache[i_0]
        F2 = fcache[j_0]
        s = y1 * y2

        if s == -1:
            L = max(0, alph2 - alph1)
            H = min(C, C + alph2 - alph1)
        else:
            L = max(0, alph2 + alph1 - C)
            H = min(C, alph2 + alph1)



        if kernel_identifier == 'standard scalar product':
            Ki0 = np.dot(data, data[i_0])
        else:
            Ki0 = np.array([kernel(data[i], data[i_0]) for i in range(l)])



        if kernel_identifier == 'standard scalar product':
            Kj0 = np.dot(data, data[j_0])
        else:
            Kj0 = np.array([kernel(data[i], data[j_0]) for i in range(l)])


        k11 = Ki0[i_0]
        k12 = Ki0[j_0]
        k22 = Kj0[j_0]

        eta = 2 * k12 - k11 - k22

        if eta < 0:
            a2 = alph2 - y2 * (F1 - F2) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            # Note that this case never occurred, so we saw no need to take care of the procedure necessary here,
            # although there is one.
            raise ValueError('Error: eta == 0')

        a1 = alph1 + s * (alph2 - a2)

        # update fcache
        fac_i_0 = y1 * (a1 - alph1)
        fac_j_0 = y2 * (a2 - alph2)
        fcache = fcache + fac_i_0 * Ki0 + fac_j_0 * Kj0

        # update alpha, I_up, I_low
        alpha[i_0] = a1
        alpha[j_0] = a2
        I[:, i_0] = I_up_low_membership(a1, y1)
        I[:, j_0] = I_up_low_membership(a2, y2)

        # needed for checking if algorithm got stuck
        i_0_old = i_0
        j_0_old = j_0

        # choose i_0,j_0 for next iteration and compute b_up, b_low for these i_0, j_0

        I_up = I[0]
        I_low = I[1]
        ind_up = v[I_up]
        ind_low = v[I_low]
        i_0s = np.argmin(fcache[ind_up])

        i_0 = ind_up[i_0s]
        b_up = fcache[i_0]

        j_0s = np.argmax(fcache[ind_low])
        j_0 = ind_low[j_0s]
        b_low = fcache[j_0]

        # check if algorithm got stuck
        if i_0 == i_0_old and j_0 == j_0_old:
            if stuckcache == [i_0, j_0]:
                stuckcounter += 1
                if stuckcounter > 5000:
                    raise ValueError('Algorithm got stuck on one violating pair and was hence highly unlikely to terminate; increase "null" in smo')
            else:
                stuckcache = [i_0, j_0]
        else:
            stuckcounter = 0

        # iter += 1

    # optional safety measure in order to make sure the solution is not tol-violating
    if violationcheckyesorno == 'yes':
        b_up = float('inf')
        b_low = -b_up
        for i in range(l):
            if I[0][i] == 1 and fcache[i] < b_up:
                b_up = fcache[i]
            if I[1][i] == 1 and fcache[i] > b_low:
                b_low = fcache[i]

        if b_low - tol <= b_up:
            violationstring = 'no tol-violation'
        else:
            violationstring = 'tol-violations!!'
    else:
        violationstring = 'no violation requested'


    return {'solution': alpha, 'violationcheck': violationstring}
