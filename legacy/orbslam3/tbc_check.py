import numpy as np
from scipy.spatial.transform import Rotation as R
S='/private/tmp/claude-501/-Users-jeonghwanlee/804e23fc-90ec-480d-8da2-22a28b640e97/scratchpad'
def wahba(A,B):
    U,_,Vt=np.linalg.svd(B.T@A); D=np.diag([1,1,np.sign(np.linalg.det(U@Vt))]); return U@D@Vt
import re
for h in ('right','left'):
    o=np.loadtxt(f'{S}/orbout/v4/orb_runs/{h}_mono_r1/online.txt'); ok=o[:,1]==2
    t=(o[:,0]*1e9).astype(np.int64); Rwc=R.from_quat(o[:,8:12])
    y=open(f'{S}/orb113243/{h}/orb_lever0.yaml').read(); M=np.array([float(v) for v in re.search(r'data: \[(.*?)\]',y).group(1).split(',')]).reshape(4,4); Rbc=M[:3,:3]
    d=np.loadtxt(f'{S}/orb113243/{h}/mav0/imu0/data.csv',delimiter=',',comments='#'); ti=d[:,0].astype(np.int64)
    m=ok[:-1]&ok[1:]&(o[:-1,2]==o[1:,2]); dt=np.diff(t)/1e9
    wc=(Rwc[:-1].inv()*Rwc[1:]).as_rotvec()/dt[:,None]  # body(camera) frame angular velocity
    m&=(np.linalg.norm(wc,axis=1)>0.3)&(np.linalg.norm(wc,axis=1)<8)
    for off in (-60,-40,-20,0,20,40,60):
        tm=(t[:-1]+t[1:])//2+off*1000000; g=np.stack([np.interp(tm,ti,d[:,1+k]) for k in range(3)],1)
        r_yaml=np.median(np.linalg.norm(g[m]-wc[m]@Rbc.T,axis=1)); Rf=wahba(wc[m],g[m]); r_fit=np.median(np.linalg.norm(g[m]-wc[m]@Rf.T,axis=1))
        print(h,'extra offset ms',off,'n',m.sum(),'resid yaml R',round(r_yaml,3),'resid fit R',round(r_fit,3),'angle yaml-vs-fit deg',round(np.degrees(np.linalg.norm(R.from_matrix(Rf@Rbc.T).as_rotvec())),1))
