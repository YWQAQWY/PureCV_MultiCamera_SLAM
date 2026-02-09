/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gomez Rodriguez, Jose M.M. Montiel and Juan D. Tardos, University of Zaragoza.
* Copyright (C) 2014-2016 Raul Mur-Artal, Jose M.M. Montiel and Juan D. Tardos, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <sstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <System.h>

using namespace std;

void LoadImages(const string &strFile,
                vector<vector<string>> &vstrImageFilenames,
                vector<double> &vTimestamps);
string MakeImagePath(const string &folder, const string &name);

int main(int argc, char **argv)
{
    if(argc < 4 || argc > 10)
    {
        cerr << endl << "Usage (association.txt with absolute paths): "
             << "./multi_fisheye path_to_vocabulary path_to_settings path_to_association.txt "
             << "(trajectory_file_name) (output_dir)" << endl;
        cerr << endl << "Usage (relative filenames + folders): "
             << "./multi_fisheye path_to_vocabulary path_to_settings "
             << "path_to_cam0 path_to_cam1 path_to_cam2 path_to_cam3 path_to_list_file "
             << "(trajectory_file_name) (output_dir)" << endl;
        return 1;
    }

    string camFolders[4] = {string(), string(), string(), string()};
    string listFile;
    bool useFolders = false;
    if(argc >= 8)
    {
        useFolders = true;
        camFolders[0] = string(argv[3]);
        camFolders[1] = string(argv[4]);
        camFolders[2] = string(argv[5]);
        camFolders[3] = string(argv[6]);
        listFile = string(argv[7]);
    }
    else
    {
        listFile = string(argv[3]);
    }

    bool bFileName = (argc == 5 || argc == 9 || argc == 6 || argc == 10);
    bool bOutDir = (argc == 6 || argc == 10);
    string fileName;
    if(bFileName)
    {
        fileName = string(argv[argc - 1]);
        cout << "trajectory file: " << fileName << endl;
    }

    string outputDir;
    if(bOutDir)
    {
        outputDir = string(argv[argc - 1]);
        if(!outputDir.empty() && outputDir.back() == '/')
            outputDir.pop_back();
    }

    vector<vector<string>> vstrImageFilenames(4);
    vector<double> vTimestamps;
    LoadImages(listFile, vstrImageFilenames, vTimestamps);

    const int nImages = static_cast<int>(vTimestamps.size());
    if(nImages == 0)
    {
        cerr << "No images found in list file: " << listFile << endl;
        return 1;
    }

    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, false);
    const float imageScale = SLAM.GetImageScale();

    const int numCams = SLAM.GetNumCameras();
    const int mainCamIndex = SLAM.GetMainCamIndex();
    if(mainCamIndex < 0 || mainCamIndex >= 4)
    {
        cerr << "main_cam_index must be in [0,3], got " << mainCamIndex << endl;
        return 1;
    }
    if(numCams != 4)
    {
        cerr << "Warning: settings Camera.nCam is " << numCams << ", example expects 4." << endl;
    }

    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    double t_resize = 0.0;
    double t_track = 0.0;

    for(int ni = 0; ni < nImages; ni++)
    {
        vector<cv::Mat> vImGray(4);
        const double tframe = vTimestamps[ni];

        for(int cam = 0; cam < 4; ++cam)
        {
            const string imagePath = useFolders ? MakeImagePath(camFolders[cam], vstrImageFilenames[cam][ni])
                                                 : vstrImageFilenames[cam][ni];
            cv::Mat im = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
            if(im.empty())
            {
                cerr << endl << "Failed to load image at: " << imagePath << endl;
                return 1;
            }

            if(im.channels() == 3)
            {
                cv::cvtColor(im, vImGray[cam], cv::COLOR_BGR2GRAY);
            }
            else if(im.channels() == 4)
            {
                cv::cvtColor(im, vImGray[cam], cv::COLOR_BGRA2GRAY);
            }
            else
            {
                vImGray[cam] = im;
            }

            if(imageScale != 1.f)
            {
                int width = static_cast<int>(vImGray[cam].cols * imageScale);
                int height = static_cast<int>(vImGray[cam].rows * imageScale);
                cv::resize(vImGray[cam], vImGray[cam], cv::Size(width, height));
            }
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        SLAM.TrackMultiCamera(vImGray, tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

#ifdef REGISTER_TIMES
        t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
        SLAM.InsertTrackTime(t_track);
#endif

        const double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        vTimesTrack[ni] = static_cast<float>(ttrack);

        double T = 0.0;
        if(ni < nImages - 1)
            T = vTimestamps[ni + 1] - tframe;
        else if(ni > 0)
            T = tframe - vTimestamps[ni - 1];

        if(ttrack < T)
            usleep((T - ttrack) * 1e6);
    }

    SLAM.Shutdown();

    const string rigTrajectory = bFileName ? fileName : "RigTrajectory.txt";
    const string rigPath = outputDir.empty() ? rigTrajectory : (outputDir + "/" + rigTrajectory);
    const string mapPath = outputDir.empty() ? string("MapPoints.xyz") : (outputDir + "/MapPoints.xyz");

    SLAM.SaveRigTrajectoryTUM(rigPath);
    SLAM.SaveMapPointsXYZ(mapPath);

    return 0;
}

void LoadImages(const string &strFile,
                vector<vector<string>> &vstrImageFilenames,
                vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());
    if(!f.is_open())
    {
        cerr << "Failed to open list file: " << strFile << endl;
        return;
    }

    string line;
    while(getline(f, line))
    {
        if(line.empty() || line[0] == '#')
            continue;

        stringstream ss(line);
        double t;
        string cam0, cam1, cam2, cam3;
        ss >> t >> cam0 >> cam1 >> cam2 >> cam3;
        if(ss.fail())
            continue;

        vTimestamps.push_back(t);
        vstrImageFilenames[0].push_back(cam0);
        vstrImageFilenames[1].push_back(cam1);
        vstrImageFilenames[2].push_back(cam2);
        vstrImageFilenames[3].push_back(cam3);
    }
    f.close();
}

string MakeImagePath(const string &folder, const string &name)
{
    if(name.empty())
        return name;
    if(!name.empty() && name[0] == '/')
        return name;
    if(folder.empty())
        return name;
    return folder + "/" + name;
}
