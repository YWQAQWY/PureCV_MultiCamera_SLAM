/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
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

#include "MapPoint.h"
#include "ORBmatcher.h"

#include<mutex>

namespace ORB_SLAM3
{

namespace {
int ObservationSize(const KeyFrame* pKF)
{
    if(pKF->NLeft == -1)
        return (pKF->mnCams > 1) ? pKF->mnCams : 1;
    return 2;
}

int ObservationCamIndex(const KeyFrame* pKF, int idx)
{
    if(pKF->NLeft == -1)
    {
        if(pKF->mnCams > 1 && idx >= 0 && idx < static_cast<int>(pKF->mvKeyCamIdx.size()))
            return pKF->mvKeyCamIdx[idx];
        return 0;
    }
    return (idx < pKF->NLeft) ? 0 : 1;
}
}

long unsigned int MapPoint::nNextId=0;
mutex MapPoint::mGlobalMutex;

MapPoint::MapPoint():
    mnFirstKFid(0), mnFirstFrame(0), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL))
{
    mpReplaced = static_cast<MapPoint*>(NULL);
    mTrackCamIdx = 0;
}

MapPoint::MapPoint(const Eigen::Vector3f &Pos, KeyFrame *pRefKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap),
    mnOriginMapId(pMap->GetId())
{
    SetWorldPos(Pos);

    mNormalVector.setZero();

    mbTrackInViewR = false;
    mbTrackInView = false;

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

MapPoint::MapPoint(const double invDepth, cv::Point2f uv_init, KeyFrame* pRefKF, KeyFrame* pHostKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap),
    mnOriginMapId(pMap->GetId())
{
    mInvDepth=invDepth;
    mInitU=(double)uv_init.x;
    mInitV=(double)uv_init.y;
    mpHostKF = pHostKF;

    mNormalVector.setZero();

    // Worldpos is not set
    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

MapPoint::MapPoint(const Eigen::Vector3f &Pos, Map* pMap, Frame* pFrame, const int &idxF):
    mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
    mnBALocalForKF(0), mnFuseCandidateForKF(0),mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
    mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap), mnOriginMapId(pMap->GetId())
{
    SetWorldPos(Pos);

    Eigen::Vector3f Ow;
    if(pFrame -> Nleft == -1 || idxF < pFrame -> Nleft){
        Ow = pFrame->GetCameraCenter();
    }
    else{
        Eigen::Matrix3f Rwl = pFrame->GetRwc();
        Eigen::Vector3f tlr = pFrame->GetRelativePoseTlr().translation();
        Eigen::Vector3f twl = pFrame->GetOw();

        Ow = Rwl * tlr + twl;
    }
    mNormalVector = mWorldPos - Ow;
    mNormalVector = mNormalVector / mNormalVector.norm();

    Eigen::Vector3f PC = mWorldPos - Ow;
    const float dist = PC.norm();
    const int level = (pFrame -> Nleft == -1) ? pFrame->mvKeysUn[idxF].octave
                                              : (idxF < pFrame -> Nleft) ? pFrame->mvKeys[idxF].octave
                                                                         : pFrame -> mvKeysRight[idxF].octave;
    const float levelScaleFactor =  pFrame->mvScaleFactors[level];
    const int nLevels = pFrame->mnScaleLevels;

    mfMaxDistance = dist*levelScaleFactor;
    mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nLevels-1];

    pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

void MapPoint::SetWorldPos(const Eigen::Vector3f &Pos) {
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    mWorldPos = Pos;
}

Eigen::Vector3f MapPoint::GetWorldPos() {
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos;
}

Eigen::Vector3f MapPoint::GetNormal() {
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector;
}


KeyFrame* MapPoint::GetReferenceKeyFrame()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
}

void MapPoint::AddObservation(KeyFrame* pKF, int idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    const int obsSize = ObservationSize(pKF);
    vector<int> indexes(obsSize, -1);

    if(mObservations.count(pKF))
        indexes = mObservations[pKF];

    if(static_cast<int>(indexes.size()) < obsSize)
        indexes.resize(obsSize, -1);

    const int camIdx = ObservationCamIndex(pKF, idx);
    if(camIdx >= 0 && camIdx < obsSize)
    {
        if(indexes[camIdx] == -1)
            nObs++;
        indexes[camIdx] = idx;
    }

    mObservations[pKF]=indexes;
}

void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad=false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
        {
            vector<int> indexes = mObservations[pKF];
            for(size_t i = 0; i < indexes.size(); ++i)
            {
                if(indexes[i] != -1)
                    nObs--;
            }
            mObservations.erase(pKF);

            if(mpRefKF==pKF)
                mpRefKF=mObservations.begin()->first;

            // If only 2 observations or less, discard point
            if(nObs<=2)
                bBad=true;
        }
    }

    if(bBad)
        SetBadFlag();
}


std::map<KeyFrame*, std::vector<int>>  MapPoint::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}

int MapPoint::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

void MapPoint::SetBadFlag()
{
    map<KeyFrame*, vector<int>> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad=true;
        obs = mObservations;
        mObservations.clear();
    }
    for(map<KeyFrame*, vector<int>>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        const vector<int> &indexes = mit->second;
        for(size_t i = 0; i < indexes.size(); ++i)
        {
            const int obsIdx = indexes[i];
            if(obsIdx != -1)
                pKF->EraseMapPointMatch(obsIdx);
        }
    }

    mpMap->EraseMapPoint(this);
}

MapPoint* MapPoint::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}

void MapPoint::Replace(MapPoint* pMP)
{
    if(pMP->mnId==this->mnId)
        return;

    int nvisible, nfound;
    map<KeyFrame*,vector<int>> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        obs=mObservations;
        mObservations.clear();
        mbBad=true;
        nvisible = mnVisible;
        nfound = mnFound;
        mpReplaced = pMP;
    }

    for(map<KeyFrame*,vector<int>>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        // Replace measurement in keyframe
        KeyFrame* pKF = mit->first;

        const vector<int> &indexes = mit->second;
        if(!pMP->IsInKeyFrame(pKF))
        {
            for(size_t i = 0; i < indexes.size(); ++i)
            {
                const int obsIdx = indexes[i];
                if(obsIdx != -1)
                {
                    pKF->ReplaceMapPointMatch(obsIdx, pMP);
                    pMP->AddObservation(pKF, obsIdx);
                }
            }
        }
        else
        {
            for(size_t i = 0; i < indexes.size(); ++i)
            {
                const int obsIdx = indexes[i];
                if(obsIdx != -1)
                    pKF->EraseMapPointMatch(obsIdx);
            }
        }
    }
    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
    pMP->ComputeDistinctiveDescriptors();

    mpMap->EraseMapPoint(this);
}

bool MapPoint::isBad()
{
    unique_lock<mutex> lock1(mMutexFeatures,std::defer_lock);
    unique_lock<mutex> lock2(mMutexPos,std::defer_lock);
    lock(lock1, lock2);

    return mbBad;
}

void MapPoint::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible+=n;
}

void MapPoint::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound+=n;
}

float MapPoint::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound)/mnVisible;
}

void MapPoint::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;

    map<KeyFrame*,vector<int>> observations;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if(mbBad)
            return;
        observations=mObservations;
    }

    if(observations.empty())
        return;

    vDescriptors.reserve(observations.size());

    for(map<KeyFrame*,vector<int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        if(!pKF->isBad()){
            const vector<int> &indexes = mit->second;
            for(size_t i = 0; i < indexes.size(); ++i)
            {
                const int obsIdx = indexes[i];
                if(obsIdx != -1)
                    vDescriptors.push_back(pKF->mDescriptors.row(obsIdx));
            }
        }
    }

    if(vDescriptors.empty())
        return;

    // Compute distances between them
    const size_t N = vDescriptors.size();

    float Distances[N][N];
    for(size_t i=0;i<N;i++)
    {
        Distances[i][i]=0;
        for(size_t j=i+1;j<N;j++)
        {
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for(size_t i=0;i<N;i++)
    {
        vector<int> vDists(Distances[i],Distances[i]+N);
        sort(vDists.begin(),vDists.end());
        int median = vDists[0.5*(N-1)];

        if(median<BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();
    }
}

cv::Mat MapPoint::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

tuple<int,int> MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
    {
        const vector<int> &indexes = mObservations[pKF];
        const int leftIndex = !indexes.empty() ? indexes[0] : -1;
        const int rightIndex = (indexes.size() > 1) ? indexes[1] : -1;
        return tuple<int,int>(leftIndex,rightIndex);
    }
    return tuple<int,int>(-1,-1);
}

bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(!mObservations.count(pKF))
        return false;
    const vector<int> &indexes = mObservations[pKF];
    for(size_t i = 0; i < indexes.size(); ++i)
    {
        if(indexes[i] != -1)
            return true;
    }
    return false;
}

void MapPoint::UpdateNormalAndDepth()
{
    map<KeyFrame*,vector<int>> observations;
    KeyFrame* pRefKF;
    Eigen::Vector3f Pos;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if(mbBad)
            return;
        observations = mObservations;
        pRefKF = mpRefKF;
        Pos = mWorldPos;
    }

    if(observations.empty())
        return;

    Eigen::Vector3f normal;
    normal.setZero();
    int n=0;
    for(map<KeyFrame*,vector<int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        const vector<int> &indexes = mit->second;
        for(size_t camIdx = 0; camIdx < indexes.size(); ++camIdx)
        {
            const int obsIdx = indexes[camIdx];
            if(obsIdx == -1)
                continue;
            Eigen::Vector3f Owi = pKF->GetCameraCenterCam(static_cast<int>(camIdx));
            Eigen::Vector3f normali = Pos - Owi;
            normal = normal + normali / normali.norm();
            n++;
        }
    }

    Eigen::Vector3f PC;
    int level = 0;
    int refCamIdx = -1;
    int refObsIdx = -1;
    if(observations.count(pRefKF))
    {
        const vector<int> &refIdxs = observations[pRefKF];
        for(size_t camIdx = 0; camIdx < refIdxs.size(); ++camIdx)
        {
            if(refIdxs[camIdx] != -1)
            {
                refCamIdx = static_cast<int>(camIdx);
                refObsIdx = refIdxs[camIdx];
                break;
            }
        }
    }
    if(refCamIdx == -1)
        return;

    PC = Pos - pRefKF->GetCameraCenterCam(refCamIdx);
    const float dist = PC.norm();

    if(pRefKF->NLeft == -1){
        level = pRefKF->mvKeysUn[refObsIdx].octave;
    }
    else if(refCamIdx == 0){
        level = pRefKF->mvKeys[refObsIdx].octave;
    }
    else{
        level = pRefKF->mvKeysRight[refObsIdx - pRefKF->NLeft].octave;
    }

    //const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
    const float levelScaleFactor =  pRefKF->mvScaleFactors[level];
    const int nLevels = pRefKF->mnScaleLevels;

    {
        unique_lock<mutex> lock3(mMutexPos);
        mfMaxDistance = dist*levelScaleFactor;
        mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1];
        mNormalVector = normal/n;
    }
}

void MapPoint::SetNormalVector(const Eigen::Vector3f& normal)
{
    unique_lock<mutex> lock3(mMutexPos);
    mNormalVector = normal;
}

float MapPoint::GetMinDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f * mfMinDistance;
}

float MapPoint::GetMaxDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f * mfMaxDistance;
}

int MapPoint::PredictScale(const float &currentDist, KeyFrame* pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}

int MapPoint::PredictScale(const float &currentDist, Frame* pF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pF->mnScaleLevels)
        nScale = pF->mnScaleLevels-1;

    return nScale;
}

void MapPoint::PrintObservations()
{
    cout << "MP_OBS: MP " << mnId << endl;
    for(map<KeyFrame*,vector<int>>::iterator mit=mObservations.begin(), mend=mObservations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKFi = mit->first;
        cout << "--OBS in KF " << pKFi->mnId << " in map " << pKFi->GetMap()->GetId() << endl;
    }
}

Map* MapPoint::GetMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mpMap;
}

void MapPoint::UpdateMap(Map* pMap)
{
    unique_lock<mutex> lock(mMutexMap);
    mpMap = pMap;
}

void MapPoint::PreSave(set<KeyFrame*>& spKF,set<MapPoint*>& spMP)
{
    mBackupReplacedId = -1;
    if(mpReplaced && spMP.find(mpReplaced) != spMP.end())
        mBackupReplacedId = mpReplaced->mnId;

    mBackupObservationsId1.clear();
    mBackupObservationsId2.clear();
    // Save the id and position in each KF who view it
    for(std::map<KeyFrame*,std::vector<int> >::const_iterator it = mObservations.begin(), end = mObservations.end(); it != end; ++it)
    {
        KeyFrame* pKFi = it->first;
        if(spKF.find(pKFi) != spKF.end())
        {
            const vector<int> &indexes = it->second;
            mBackupObservationsId1[it->first->mnId] = !indexes.empty() ? indexes[0] : -1;
            mBackupObservationsId2[it->first->mnId] = (indexes.size() > 1) ? indexes[1] : -1;
        }
        else
        {
            EraseObservation(pKFi);
        }
    }

    // Save the id of the reference KF
    if(spKF.find(mpRefKF) != spKF.end())
    {
        mBackupRefKFId = mpRefKF->mnId;
    }
}

void MapPoint::PostLoad(map<long unsigned int, KeyFrame*>& mpKFid, map<long unsigned int, MapPoint*>& mpMPid)
{
    mpRefKF = mpKFid[mBackupRefKFId];
    if(!mpRefKF)
    {
        cout << "ERROR: MP without KF reference " << mBackupRefKFId << "; Num obs: " << nObs << endl;
    }
    mpReplaced = static_cast<MapPoint*>(NULL);
    if(mBackupReplacedId>=0)
    {
        map<long unsigned int, MapPoint*>::iterator it = mpMPid.find(mBackupReplacedId);
        if (it != mpMPid.end())
            mpReplaced = it->second;
    }

    mObservations.clear();

    for(map<long unsigned int, int>::const_iterator it = mBackupObservationsId1.begin(), end = mBackupObservationsId1.end(); it != end; ++it)
    {
        KeyFrame* pKFi = mpKFid[it->first];
        map<long unsigned int, int>::const_iterator it2 = mBackupObservationsId2.find(it->first);
        vector<int> indexes;
        indexes.push_back(it->second);
        indexes.push_back(it2->second);
        if(pKFi)
        {
           mObservations[pKFi] = indexes;
        }
    }

    mBackupObservationsId1.clear();
    mBackupObservationsId2.clear();
}

} //namespace ORB_SLAM
