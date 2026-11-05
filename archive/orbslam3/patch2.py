R='/workspace/ORB_SLAM3/'
f=R+'src/Frame.cc'; c=open(f).read()
if 'ORB_MASK' not in c:
    anchor='    ExtractORB(0,imGray,0,1000);\n'
    assert c.count(anchor)==1
    c=c.replace(anchor, anchor+'''    { static cv::Mat sMask; static bool sTried=false;
      if(!sTried){ sTried=true; const char* mp=getenv("ORB_MASK"); if(mp) sMask=cv::imread(mp,cv::IMREAD_GRAYSCALE); }
      if(!sMask.empty() && !mvKeys.empty()){
        float sx=(float)sMask.cols/imGray.cols, sy=(float)sMask.rows/imGray.rows; std::vector<cv::KeyPoint> k2; cv::Mat d2;
        for(size_t i=0;i<mvKeys.size();i++){ int u=std::min(sMask.cols-1,std::max(0,(int)(mvKeys[i].pt.x*sx))), v=std::min(sMask.rows-1,std::max(0,(int)(mvKeys[i].pt.y*sy)));
          if(sMask.at<uchar>(v,u)==0){ k2.push_back(mvKeys[i]); d2.push_back(mDescriptors.row(i)); } }
        mvKeys=k2; mDescriptors=d2; } } // ORB_MASK
''',1)
    if '#include <cstdlib>' not in c: c='#include <cstdlib>\n'+c
    open(f,'w').write(c)
f=R+'src/Tracking.cc'; c=open(f).read()
if 'ORB_KEEP' not in c:
    a='''                    cout << "IMU is not or recently initialized. Reseting active map..." << endl;
                    mpSystem->ResetActiveMap();'''
    assert c.count(a)==1
    c=c.replace(a,'''                    if(getenv("ORB_KEEP") && pCurrentMap->KeyFramesInMap()>10) { cout << "KEEP: lost before IMU BA2, new map instead of reset" << endl; mState=LOST; }
                    else { cout << "IMU is not or recently initialized. Reseting active map..." << endl;
                    mpSystem->ResetActiveMap(); } // ORB_KEEP''')
    b='''                if (!pCurrentMap->isImuInitialized())
                {
                    Verbose::PrintMess("Track lost before IMU initialisation, reseting...", Verbose::VERBOSITY_QUIET);'''
    assert c.count(b)==1
    c=c.replace(b,'''                if (!pCurrentMap->isImuInitialized() && !getenv("ORB_KEEP"))
                {
                    Verbose::PrintMess("Track lost before IMU initialisation, reseting...", Verbose::VERBOSITY_QUIET);''')
    if '#include <cstdlib>' not in c: c='#include <cstdlib>\n'+c
    open(f,'w').write(c)
e=R+'Examples/Monocular-Inertial/mono_inertial_euroc.cc'; x=open(e).read()
if 'ALLMAPS_FIRST' not in x:
    x=x.replace('        SLAM.SaveTrajectoryAllMaps("allmaps.txt");\n','')
    a='        SLAM.SaveTrajectoryEuRoC(f_file);\n        SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);\n'
    assert x.count(a)==1
    x=x.replace(a,'        SLAM.SaveTrajectoryAllMaps("allmaps.txt"); // ALLMAPS_FIRST\n')
    open(e,'w').write(x)
print('patch2 ok')
