from jax.scipy.special import betainc

def quantify_risk(k,N,beta):
    """
        calculates the upper and lower bounds on the risk given:
        N (int) - number of samples
        k (int) - length of support list
        beta (float) - confidence
        NOTE: betainc is the incomplete beta function, a naming coincidence with beta the confidence parameter in this approach
                compared with MATLAB, betainc has the first argument as the last argument instead.
        Outputs:
        epsL (float) - lower bound of the risk
        epsU (float) - upper bound of the risk
    """
    t1 = 0.0
    t2 = k/N
    threshold = 1e-10


    while (t2 - t1) > threshold:
        t = (t1+t2)/2
        left = beta/3*betainc(k+1,N-k,t)+beta/6*betainc(k+1,4*N+1-k,t)
        #print("left = ",left)
        right = (1+beta/6/N)*t*N*(betainc(k,N-k+1,t)-betainc(k+1,N-k,t))
        #print("right= ", right)
        if left > right: #added threshold
            t1 = t
        else:
            t2 = t
    epsL = t1 #set lower bound
    #print("epsL = ", epsL)

    if (k==N):
        epsU = 1 #set upper bound
    else:
        t1 = k/N
        t2 = 1
        while (t2 - t1) > threshold:
            t = (t1 + t2) / 2
            left = (beta / 2 - beta / 6) * betainc(k + 1, N - k,t) + beta / 6 * betainc(k + 1, 4 * N + 1 - k,t)
            #print(left)
            right = (1 + beta / 6 / N) * t * N * (betainc( k, N - k + 1,t) - betainc( k + 1, N - k,t))
            #print(right)
            if left > right:
                t2 = t
            else:
                t1 = t
        epsU = t2 #set upper bound
    #print("epsU = ", epsU)
    # Return results
    return epsL, epsU

def quantify_conf():
    """
        To create...
    """


    # Return results
    return 0.0