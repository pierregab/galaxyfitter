import numpy as np

def density_and_slope(r,M0,Rs,rc,n):
    r=np.asarray(r);x=r/Rs;u=r/rc;t=np.tanh(u);dt=1-t*t
    mass=M0*(np.log1p(x)-x/(1+x))
    a=M0/(4*np.pi*Rs**3)/(x*(1+x)**2)*t**n
    b=n*t**(n-1)*dt*mass/(4*np.pi*r*r*rc)
    da=-1-2*x/(1+x)+n*u*dt/t
    db=M0*x*x/((1+x)**2*mass)+(n-1)*u*dt/t-2*u*t-2
    rho=a+b
    assert np.all(rho>0) and np.isfinite(rho).all()
    return rho,(a*da+b*db)/rho

def profiles(scan,r):
    return np.array([density_and_slope(r,q.M0,q.Rs,q.r_core,q.n_core)[1] for q in scan.itertuples()])
