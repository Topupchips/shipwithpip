import numpy as np
from collections import deque
G=100; STEPS=200; N=120
WALLS=[(0,46,72,50)]  # x0,y0,x1,y1 (y down)
START=np.array([50.0,94.0]); GOAL=np.array([50.0,8.0]); GR=5.0
def in_wall(x,y):
    for x0,y0,x1,y1 in WALLS:
        if x0<=x<=x1 and y0<=y<=y1: return True
    return x<0 or x>G or y<0 or y>G
def dist_field():
    d=np.full((G+1,G+1),1e9); q=deque()
    gx,gy=int(GOAL[0]),int(GOAL[1]); d[gy,gx]=0; q.append((gx,gy))
    while q:
        x,y=q.popleft()
        for dx,dy in((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)):
            nx,ny=x+dx,y+dy
            if 0<=nx<=G and 0<=ny<=G and not in_wall(nx,ny) and d[ny,nx]>d[y,x]+(1.414 if dx and dy else 1):
                d[ny,nx]=d[y,x]+(1.414 if dx and dy else 1); q.append((nx,ny))
    return d
DF=dist_field()
def run(genomes):
    n=len(genomes); pos=np.tile(START,(n,1)); vel=np.zeros((n,2)); alive=np.ones(n,bool); reached=np.zeros(n,bool); rstep=np.full(n,STEPS); dstep=np.full(n,STEPS)
    hist=np.zeros((STEPS+1,n,2)); st=np.zeros((STEPS+1,n),np.int8); hist[0]=pos
    for s in range(STEPS):
        a=genomes[:,s]
        acc=np.stack([np.cos(a),np.sin(a)],1)*0.22
        vel=(vel+acc)*0.96; sp=np.linalg.norm(vel,axis=1,keepdims=True); vel=np.where(sp>1.5,vel/sp*1.5,vel)
        new=pos+vel
        for i in range(n):
            if not alive[i]: continue
            if in_wall(new[i,0],new[i,1]): alive[i]=False; dstep[i]=s; continue
            pos[i]=new[i]
            if np.linalg.norm(pos[i]-GOAL)<GR: alive[i]=False; reached[i]=True; rstep[i]=s
        hist[s+1]=pos; st[s+1]=np.where(reached,2,np.where(alive,1,0))
    return hist,st,reached,rstep,pos,dstep
def fitness(pos,reached,rstep):
    f=np.array([-DF[int(np.clip(p[1],0,G)),int(np.clip(p[0],0,G))] for p in pos])
    return np.where(reached,1000+(STEPS-rstep),f)
def evolve(seed=3,maxg=60,target=0.6):
    rng=np.random.default_rng(seed)
    gen=rng.uniform(-np.pi,np.pi,(N,STEPS)); out=[]
    for g in range(maxg):
        hist,st,reached,rstep,pos,dstep=run(gen); fit=fitness(pos,reached,rstep)
        out.append(dict(gen=g+1,hist=hist.astype(np.float32),st=st,reached=int(reached.sum()),best=float(-np.max(np.where(reached,0,fit)) ) if (~reached).any() else 0.0,fit=fit,bi=int(np.argmax(fit))))
        if reached.sum()>=target*N: break
        order=np.argsort(-fit); ne=max(4,N//10); new=[gen[i] for i in order[:ne]]
        top=order[:N//3]
        while len(new)<N:
            a,b=rng.choice(top,2,replace=False); cut=rng.integers(20,STEPS-20)
            c=np.concatenate([gen[a][:cut],gen[b][cut:]])
            dp=dstep[a] if rng.random()<0.5 else dstep[b]
            rate=np.where(np.arange(STEPS)>=dp-45,0.28,0.04); m=rng.random(STEPS)<rate; c=np.where(m,c+rng.normal(0,1.1,STEPS),c)
            new.append(c)
        gen=np.array(new)
    return out
if __name__=="__main__":
    import sys
    for seed in range(1,9):
        o=evolve(seed); print(seed,len(o),[x['reached'] for x in o])
