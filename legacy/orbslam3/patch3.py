R='/workspace/ORB_SLAM3/'; f=R+'src/LocalMapping.cc'; c=open(f).read()
if 'ORB_PRIORA' not in c:
    c=c.replace('InitializeIMU(1e2, 1e10, true);','InitializeIMU(1e2, getenv("ORB_PRIORA")?atof(getenv("ORB_PRIORA")):1e10, true); // ORB_PRIORA')
    c=c.replace('InitializeIMU(1.f, 1e5, true);','InitializeIMU(1.f, getenv("ORB_PRIORA")?atof(getenv("ORB_PRIORA")):1e5, true);')
    c='#include <cstdlib>\n'+c; open(f,'w').write(c)
print('patch3 ok')
