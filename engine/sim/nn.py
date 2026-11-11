import numpy as np
def spiral(n=200,seed=0):
    r=np.random.default_rng(seed); X=[];y=[]
    for c in (0,1):
        t=np.linspace(0.25,1,n); ang=t*3.6*np.pi+c*np.pi; rad=t
        X.append(np.stack([rad*np.cos(ang),rad*np.sin(ang)],1)+r.normal(0,0.02,(n,2))); y+= [c]*n
    return np.concatenate(X),np.array(y,float)
def train(epochs=2500,seed=1,snap=None,H=32,lr=0.02):
    X,y=spiral(); r=np.random.default_rng(seed)
    P=[r.normal(0,0.6,(2,H)),np.zeros(H),r.normal(0,0.35,(H,H)),np.zeros(H),r.normal(0,0.35,(H,1)),np.zeros(1)]
    m=[np.zeros_like(p) for p in P]; v=[np.zeros_like(p) for p in P]
    def fwd(X):
        h1=np.tanh(X@P[0]+P[1]); h2=np.tanh(h1@P[2]+P[3]); z=(h2@P[4]+P[5]).ravel(); return h1,h2,1/(1+np.exp(-z))
    snaps=[]; gx,gy=np.meshgrid(np.linspace(-1.15,1.15,96),np.linspace(-1.15,1.15,96)); grid=np.stack([gx.ravel(),gy.ravel()],1)
    for e in range(epochs+1):
        h1,h2,p=fwd(X); loss=-np.mean(y*np.log(p+1e-9)+(1-y)*np.log(1-p+1e-9)); acc=np.mean((p>0.5)==y)
        if snap is None or e in snap: snaps.append((e,loss,acc,fwd(grid)[2].reshape(96,96)))
        d=(p-y)[:,None]/len(y); gW3=h2.T@d; gb3=d.sum(0); d2=(d@P[4].T)*(1-h2**2); gW2=h1.T@d2; gb2=d2.sum(0); d1=(d2@P[2].T)*(1-h1**2); gW1=X.T@d1; gb1=d1.sum(0)
        g=[gW1,gb1,gW2,gb2,gW3,gb3]
        for i in range(6):
            m[i]=0.9*m[i]+0.1*g[i]; v[i]=0.999*v[i]+0.001*g[i]**2
            P[i]-=lr*(m[i]/(1-0.9**(e+1)))/(np.sqrt(v[i]/(1-0.999**(e+1)))+1e-8)
    return X,y,snaps
if __name__=="__main__":
    X,y,s=train(3000,snap={0,50,100,200,400,800,1200,1600,2000,2500,3000})
    print([(e,round(l,3),round(a,3)) for e,l,a,_ in s])
