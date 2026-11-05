import numpy as np, sys
from scipy.spatial.transform import Rotation as R
S='/private/tmp/claude-501/-Users-jeonghwanlee/804e23fc-90ec-480d-8da2-22a28b640e97/scratchpad'
T='/Users/jeonghwanlee/ego_collector/datasets/tag_rig/20261021_113243'
def wahba(A,B):  # find Rm: B ~ Rm A
    U,_,Vt=np.linalg.svd(B.T@A); D=np.diag([1,1,np.sign(np.linalg.det(U@Vt))]); return U@D@Vt
for h in ('right','left'):
    z=np.load(f'{T}/umi_traj_{h}.npz'); tag=z['ntags']>0; t=z['t_ns'][tag].astype(np.int64); Rb=R.from_matrix(z['R_B_tcp'][tag]); pb=z['p_B'][tag]
    d=np.loadtxt(f'{S}/orb113243/{h}/mav0/imu0/data.csv',delimiter=',',comments='#'); ti=d[:,0].astype(np.int64)
    best=None
    for off in range(-120,121,10):
        # tag angular velocity in tcp frame from consecutive tag frames
        dtt=np.diff(t)/1e9; ok=(dtt>0.02)&(dtt<0.07)
        rv=(Rb[:-1].inv()*Rb[1:]).as_rotvec()/dtt[:,None]; tm=(t[:-1]+t[1:])//2 + off*1000000
        g=np.stack([np.interp(tm,ti,d[:,1+k]) for k in range(3)],1)
        m=ok&(np.linalg.norm(rv,axis=1)>0.3)&(np.linalg.norm(rv,axis=1)<6)
        Rm=wahba(rv[m],g[m]); res=np.linalg.norm(g[m]-rv[m]@Rm.T,axis=1)
        if best is None or np.median(res)<best[0]: best=(np.median(res),off,Rm,m.sum())
    res,off,Rm,n=best; print(h,'gyro: best IMU-vs-tag offset ms',off,'median resid rad/s',round(res,3),'n',n)
    # accel: still frames -> specific force ~ +g up. a_imu = Rm @ R_B_tcp^T @ (9.81*u)
    a=np.stack([np.interp(t+off*1000000,ti,d[:,4+k]) for k in range(3)],1)
    w=np.stack([np.interp(t+off*1000000,ti,d[:,1+k]) for k in range(3)],1)
    still=np.linalg.norm(w,axis=1)<0.1
    # vectors in base frame implied by accel: v_B = R_B_tcp @ Rm^T @ a
    vB=Rb[still].apply(a[still]@Rm); u=vB.mean(0); u/=np.linalg.norm(u)
    ang=np.degrees(np.arccos(np.clip((vB/np.linalg.norm(vB,axis=1,keepdims=True))@u,-1,1)))
    print('  accel (gyro-frame rotation) gravity-in-base consistency: n',still.sum(),'median dev deg',np.median(ang).round(1),'p90',np.percentile(ang,90).round(1),'u',u.round(3))
    # free fit of accel rotation
    # find Ra s.t. R_B_tcp Ra^T a = const: alternate
    Ra=Rm.copy()
    for it in range(20):
        vB=Rb[still].apply(a[still]@Ra); u=vB.mean(0); u/=np.linalg.norm(u)
        tgt=Rb[still].inv().apply(np.tile(u*9.81,(still.sum(),1))); Ra=wahba(tgt,a[still])
    vB=Rb[still].apply(a[still]@Ra); ang2=np.degrees(np.arccos(np.clip((vB/np.linalg.norm(vB,axis=1,keepdims=True))@u,-1,1)))
    print('  accel own best rotation: median dev', np.median(ang2).round(1),'angle between accel-frame and gyro-frame rotations deg', np.degrees(np.linalg.norm(R.from_matrix(Ra@Rm.T).as_rotvec())).round(1))
    # orientation spread at still frames
    print('  still tcp-orientation spread deg p90', np.percentile(np.degrees(np.linalg.norm((Rb[still][0].inv()*Rb[still]).as_rotvec(),axis=1)),90).round(1))
