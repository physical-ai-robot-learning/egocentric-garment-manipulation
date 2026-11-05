import re,sys
R='/workspace/ORB_SLAM3/'
h=open(R+'include/System.h').read()
if 'SaveTrajectoryAllMaps' not in h:
    h=h.replace('    void SaveTrajectoryEuRoC(const string &filename);',
     '    void SaveTrajectoryEuRoC(const string &filename);\n    void SaveTrajectoryAllMaps(const string &filename);\n    int CurMapId();\n    int CurMapImuInit();\n    int NumMaps();',1)
    assert 'SaveTrajectoryAllMaps' in h; open(R+'include/System.h','w').write(h)
c=open(R+'src/System.cc').read()
if 'SaveTrajectoryAllMaps' not in c:
    i=c.rfind('} //namespace'); assert i>0; c=c[:i]
    c+='''
int System::CurMapId(){ Map* m=mpAtlas->GetCurrentMap(); return m? (int)m->GetId() : -1; }
int System::CurMapImuInit(){ Map* m=mpAtlas->GetCurrentMap(); return (m && m->isImuInitialized())?1:0; }
int System::NumMaps(){ return (int)mpAtlas->GetAllMaps().size(); }
void System::SaveTrajectoryAllMaps(const string &filename)
{
    // per frame: t_s map_id lost twc(3) qwc(xyzw) ; camera pose in its (final) map world, after all BA / merges
    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    set<Map*> alive(vpMaps.begin(), vpMaps.end());
    ofstream f; f.open(filename.c_str()); f << fixed;
    auto lRit = mpTracker->mlpReferences.begin(); auto lT = mpTracker->mlFrameTimes.begin(); auto lbL = mpTracker->mlbLost.begin();
    for(auto lit=mpTracker->mlRelativeFramePoses.begin(); lit!=mpTracker->mlRelativeFramePoses.end(); lit++, lRit++, lT++, lbL++)
    {
        if(*lbL) { f << setprecision(9) << *lT << " -1 1 0 0 0 0 0 0 1" << endl; continue; }
        KeyFrame* pKF = *lRit; Sophus::SE3f Trw; int ok=1;
        if(!pKF) ok=0;
        int guard=0;
        while(ok && pKF->isBad()) { Trw = Trw * pKF->mTcp; pKF = pKF->GetParent(); if(!pKF || ++guard>100000) ok=0; }
        if(ok && alive.find(pKF->GetMap())==alive.end()) ok=0;
        if(!ok) { f << setprecision(9) << *lT << " -1 1 0 0 0 0 0 0 1" << endl; continue; }
        Trw = Trw * pKF->GetPose();
        Sophus::SE3f Tcw = (*lit) * Trw; Sophus::SE3f Twc = Tcw.inverse();
        Eigen::Vector3f t = Twc.translation(); Eigen::Quaternionf q = Twc.unit_quaternion();
        f << setprecision(9) << *lT << " " << pKF->GetMap()->GetId() << " 0 " << t(0) << " " << t(1) << " " << t(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
    }
    f.close();
    cout << endl << "all-maps trajectory saved, maps=" << vpMaps.size() << endl;
}

} //namespace ORB_SLAM
'''
    open(R+'src/System.cc','w').write(c)
e=R+'Examples/Monocular-Inertial/mono_inertial_euroc.cc'; x=open(e).read()
x=x.replace('ORB_SLAM3::System::IMU_MONOCULAR, true)','ORB_SLAM3::System::IMU_MONOCULAR, false)')
if 'online_' not in x:
    x=x.replace('    // Main loop','    std::ofstream fon("online.txt"); fon << std::fixed;\n    // Main loop',1)
    x=x.replace('            SLAM.TrackMonocular(im,tframe,vImuMeas); // TODO change to monocular_inertial',
     '            Sophus::SE3f Tcw_on = SLAM.TrackMonocular(im,tframe,vImuMeas);\n            { Sophus::SE3f Twc_on=Tcw_on.inverse(); Eigen::Vector3f tt=Twc_on.translation(); Eigen::Quaternionf qq=Twc_on.unit_quaternion();\n              fon << std::setprecision(9) << tframe << " " << SLAM.GetTrackingState() << " " << SLAM.CurMapId() << " " << SLAM.CurMapImuInit() << " " << SLAM.NumMaps() << " " << tt(0) << " " << tt(1) << " " << tt(2) << " " << qq.x() << " " << qq.y() << " " << qq.z() << " " << qq.w() << "\\n"; } // online_',1)
    assert 'online_' in x
    x=x.replace('        SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);','        SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);\n        SLAM.SaveTrajectoryAllMaps("allmaps.txt");',1)
    assert 'allmaps.txt' in x
    if '#include <iomanip>' not in x: x='#include <iomanip>\n'+x
    open(e,'w').write(x)
print('patched')
