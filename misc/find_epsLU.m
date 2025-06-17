function [epsL, epsU] = find_epsLU(k,N,delta)
t1 = 0;
t2 = k/N;
while t2-t1 > 1e-10
    t = (t1+t2)/2;
    left = delta/3*betainc(t,k+1,N-k)+delta/6*betainc(t,k+1,4*N+1-k);
    right = (1+delta/6/N)*t*N*(betainc(t,k,N-k+1)-betainc(t,k+1,N-k));
    if left > right
        t1=t;
    else
        t2=t;
    end
end
epsL = t1;
if k==N
    epsU = 1;
else
    t1 = k/N;
    t2 = 1;
    while t2-t1 > 1e-10
        t = (t1+t2)/2;
        left = (delta/2-delta/6)*betainc(t,k+1,N-k)+delta/6*betainc(t,k+1,4*N+1-k);
        right = (1+delta/6/N)*t*N*(betainc(t,k,N-k+1)-betainc(t,k+1,N-k));
        if left > right
            t2=t;
        else
            t1=t;
        end
    end
    epsU = t2;
end