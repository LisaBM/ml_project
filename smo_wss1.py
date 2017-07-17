import numpy as np


def smo(data, label, C, kernel, tol, violationcheckyesorno,kernel_identifier = None):
    def I_up_low_membership(alpha_i, label_i):

        # very important!
        null = 1e-16
        v = np.array([False, False])
        if (alpha_i < C - null and label_i == 1) or (alpha_i > null and label_i == -1):
            v[0] = True
        if (alpha_i < C - null and label_i == -1) or (alpha_i > null and label_i == 1):
            v[1] = True
        return v

    # data: images are rows of array, so the number of columns is 28**2 and the number of rows is the number of training data
    l = label.shape[0]
    alpha = np.zeros(l)

    I = np.zeros((2, l), dtype=bool)

    # initialize I, now logical vector! I[0] = I_up, I[1] = I_low
    I[0] = label == 1
    I[1] = label == -1

    b_up = -1
    b_low = 1

    v = np.array(range(l))
    i_up_array = v[I[0]]
    i_low_array = v[I[1]]

    i_0 = i_up_array[0]
    j_0 = i_low_array[0]

    fcache = -label.astype(float)

    # iter = 0
    # cycle = 0
    stuckcache = -np.ones(2)
    stuckcounter = 0

    # kostenintensiv bei SMO mit maximal violating pairs ist das ständige Neuberechnen der kompletten Kernel-Matrix;
    # initialisiere daher lxl - Nullmatrix und speichere alle bisher berechneten kernel-Berechnung ab
    # die Idee ist, dass das funktionieren sollte, da die meisten alpha_i auf Null bleiben; also sollte auch unsere Gram-Matrix
    # sparse bleiben

    # sparse, i.e., no memory overflow so far
    K = np.empty([l, l])

    rows_calc = []

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

        if i_0 not in rows_calc:
            if kernel_identifier == 'standard scalar product':
                K[i_0] = np.dot(data, data[i_0])
            else:
                K[i_0] = [kernel(data[i], data[i_0]) for i in range(l)]
            rows_calc.append(i_0)

        if j_0 not in rows_calc:
            if kernel_identifier == 'standard scalar product':
                K[j_0] = np.dot(data, data[j_0])
            else:
                K[j_0] = [kernel(data[i], data[j_0]) for i in range(l)]
            rows_calc.append(j_0)

        k11 = K[i_0, i_0]
        k12 = K[i_0, j_0]
        k22 = K[j_0, j_0]

        eta = 2 * k12 - k11 - k22

        if eta < 0:
            a2 = alph2 - y2 * (F1 - F2) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            # print('Error: eta == 0')
            raise ValueError('Error: eta == 0')

        a1 = alph1 + s * (alph2 - a2)

        # zum Vergleich, siehe unten, wo untersucht wird, wie oft es vorkommt, dass
        # ein Paar nach einem Schritt sofort wieder violating ist
        # alpha_old = np.empty(alpha.shape)
        # np.copyto(alpha_old, alpha)

        fac_i_0 = y1 * (a1 - alph1)
        fac_j_0 = y2 * (a2 - alph2)
        fcache = fcache + fac_i_0 * K[i_0] + fac_j_0 * K[j_0]

        # update alpha, I_up, I_low
        alpha[i_0] = a1
        alpha[j_0] = a2

        I[:, i_0] = I_up_low_membership(a1, y1)
        I[:, j_0] = I_up_low_membership(a2, y2)

        # 4.) berechne neues i_0 und j_0  für maximally violating pair und dazu b_up, b_low


        # now choose i_0,j_0 for next iteration and compute b_up, b_low for these i_0,j_0


        I_up = I[0]
        I_low = I[1]

        i_0_old = i_0
        j_0_old = j_0
        b_up = float('inf')
        b_low = -b_up

        for i in range(l):
            if I_up[i] == 1 and fcache[i] < b_up:
                b_up = fcache[i]
                i_0 = i
            if I_low[i] == 1 and fcache[i] > b_low:
                b_low = fcache[i]
                j_0 = i

        if i_0 == i_0_old and j_0 == j_0_old:
            if stuckcache == [i_0, j_0]:
                stuckcounter += 1
                if stuckcounter > 5000:
                    raise ValueError('Algorithm got stuck on one violating pair and was hence highly unlikely to terminate; increase "null" in smo')
            else:
                stuckcache = [i_0, j_0]
        else:
            stuckcounter = 0


                # if i_0_old == i_0 and j_0_old == j_0:
                # cycle += 1
                # print('Achtung, Paar zweimal hintereinander violating:')
                # print('i_0 =',i_0,'j_0 =',j_0)
                # print('alpha[i_0] =',alpha[i_0], 'alpha[j_0] =',alpha[j_0])
                # print('alpha[i_0]_old =',alph1, 'alpha[j_0]_old =',alph2)
                # print('fcache[i_0] =',fcache[i_0], 'fcache[j_0] =', fcache[j_0])
                # print('fcache[i_0]_old =',F(i_0,alpha_old), 'fcache[j_0]_old =', F(j_0,alpha_old))
                # print('b_up - b_low =', b_up - b_low)

                # iter += 1

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

        # print('Anzahl wiederholt dasselbe Paar =', cycle)
    # print('iter =', iter)

    return {'solution': alpha, 'violationcheck': violationstring}
