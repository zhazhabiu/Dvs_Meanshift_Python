import numpy as np
from meanshift import *
from Kalman_filter import createKalmanFilter
np.random.seed(0)

colormap = ['c', 'g', 'r', 'm', 'y', 'b', 'w', 'k']
shapemap = ['o', '1', 's', 'p', '+', '*', 'x', 'v']
    
class MultiTrack():
    def __init__(self, stream, size=(720, 1280), max_clusters=256):
        self.events = stream # x, y, t, p
        self.h = size[0]
        self.w = size[1]
        self.packet = 1800
        self.bandwidth = 0.15 # meanshift bandwidth
        self.max_clusters = max_clusters
        self.lastClusterColor = 0
        self.activateEvents = np.zeros(self.events.shape[0], np.bool8)
        self.reset_variants()
        self.save_allTrajectories = []

    def reset_variants(self):
        # kalman filter of all clusters
        self.kfilters = []
        # state vectors of all clusters
        self.state_vecs = []
        # measurement vectors of all clusters
        self.measure_vecs = []
        # cluster previous tick timestamp
        self.prevticks = np.zeros(self.max_clusters)
        # each value represents the number of cluster centers belonging to the existing trajectory 
        self.counterTrajectories = np.zeros(self.max_clusters)
        self.foundBlobs = np.zeros(self.max_clusters, np.bool8)
        self.foundTrajectory = np.zeros(self.max_clusters, np.bool8)
        self.lastClusterColor = 0
    
    def assignClusterCOlor(self, numClusters, matches, prev_positionClusterColor, activeTrajectories):
        """
        assign new colors to clusters
        Input:
            numClusters: number of clusters
            matches(list([])): matching ids between current clusters and previous clusters
            activeTrajectories(numpy.array[max_cluster])
        Returns:
            positionClusterColor: corresponding matching color id (if not exist, assign a new color),index-current cluster;value-previous cluster color
            存在一个性质，除了已有的轨迹，新增的轨迹id必定有序且为升序
        """
        positionClusterColor = []
        if len(matches)==0: # no mask --> assing all new colors
            for i in range(numClusters):
                positionClusterColor.append(self.lastClusterColor)
                self.lastClusterColor += 1
        else:
            for i in range(numClusters):
                if matches[i]==-1:
                    while ((self.lastClusterColor & (self.max_clusters-1)) in activeTrajectories): # Check if the new position is being used
                        self.lastClusterColor += 1
                    positionClusterColor.append(self.lastClusterColor)
                    self.lastClusterColor += 1
                else: # Match with previous cluster --> assign color of previous cluster
                    positionClusterColor.append(prev_positionClusterColor[matches[i]])
        return positionClusterColor
    
    def findClosestCluster(self, cluster1_center, cluster2_center):
        """
        cluster1 : current
        cluster2 : prev
        find clusters matching between current(cluster1) window and previous(cluster2) window with L2 distance
        """
        if cluster2_center.shape[0]==0:
            return np.full(cluster1_center.shape[0], -1)
        dist = np.zeros(len(cluster1_center))
        matches = []
        for i in range(len(cluster1_center)):
            dist[i] = 500
            matches.append(-1)
            for j in range(len(cluster2_center)):
                tmp = (cluster1_center[i, 0]*self.w-cluster2_center[j, 0]*self.w)**2  + \
                    (cluster1_center[i, 1]*self.h-cluster2_center[j, 1]*self.h)**2
                if tmp < dist[i]:
                    dist[i] = tmp
                    matches[i] = j
        return matches
    
    def forward(self, savedir='./tracking_res/'):
        TSmap = np.zeros((self.h, self.w), np.float32) # recording the lastest event timestamp
        cnt = 0
        usTime = 10.0 # in us
        firsttimestamp = 1e-6*self.events[10, 2]
        final_timestamp = 1e-6*self.events[-1, 2]
        activeEvents = np.zeros(len(self.events), np.bool8)
        prev_activeTrajectories = []
        prevclusterCenter = np.empty((0, 3))
        prev_positionClusterColor = np.array([])
        counterIn = 0
        for start in range(0, len(self.events), self.packet):
            counterOut = 0
            ssize = min(self.packet, len(self.events)-start)
            data = np.zeros((ssize, 3), np.float32)
            for i in range(start, start+ssize):
                x, y, _, _ = self.events[i]
                event_timestamp = 1E-6*self.events[counterIn, 2]-firsttimestamp # now in usecs
                ts = 1E-6*self.events[i, 2] - firsttimestamp
                maxTs = -1
                # searching the max timestamp in 3x3 neighbor
                TSmap[y, x] = 0
                for xx in range(-1, 2):
                    for yy in range(-1, 2):
                        posx = x + xx
                        posy = y + yy
                        if(posx < 0):
                            posx = 0
                        if(posy < 0):
                            posy = 0
                        if(posx > self.w-1):
                            posx = self.w-1
                        if(posy > self.h-1):
                            posy = self.h-1

                        if(TSmap[posy, posx] > maxTs):
                            maxTs = TSmap[posy, posx]
                TSmap[y, x] = ts
                # filter the event that the difference between its timestamp and its neighbor biggest timestamp > usTime   
                # 带有时空领域滤波的效果：排除孤立事件   
                if TSmap[y, x] >= maxTs + usTime:
                    activeEvents[counterIn] = False
                    counterIn += 1
                else:
                    data[counterOut, 0] = x/self.w
                    data[counterOut, 1] = y/self.h
                    tau = 1e+4
                    data[counterOut, 2] = np.exp((final_timestamp-event_timestamp)/tau) # exponential decay function
                    self.activateEvents[counterIn] = True
                    counterOut += 1
                    counterIn += 1
            last_timestamp = 1E-6*self.events[counterIn-1, 2]
            data = data[:counterOut]
            # events cluster & match
            clusterCenter, p2Cluster = meanShiftCluster_Gaussian(data, self.bandwidth)
            
            if clusterCenter.size == 0:
                print('Cluster no results in current window.')
                self.reset_variants()
                continue
            
            print(f'cluster number {clusterCenter.shape[0]}')
            matching_id = self.findClosestCluster(clusterCenter[:, :2], prevclusterCenter[:, :2])
            
            positionClusterColor = self.assignClusterCOlor(clusterCenter.shape[0], matching_id, prev_positionClusterColor, prev_activeTrajectories)
            
            activeTrajectories = []
            # cluster tracking
            if len(p2Cluster) > 400: 
                for i in range(clusterCenter.shape[0]): 
                    """
                    positionClusterColor[i]: represents the id of tracjectory; can be index to fixed array
                    trajectory_index: represents the index of trajectory list; can be index to dynamic list
                    prevTrajectory[trajectory_index] == positionClusterColor[i]
                    """
                    tmpColorPos = (positionClusterColor[i])&(self.max_clusters-1)
                    if tmpColorPos not in prev_activeTrajectories:
                        # if the position color is not found in tracks
                        # Initialize a new trajectory
                        if tmpColorPos < len(self.kfilters):
                            self.kfilters[tmpColorPos] = createKalmanFilter(self.kfilters[tmpColorPos])
                        else:
                            self.kfilters.append(createKalmanFilter())
                            self.state_vecs.append(np.zeros(4, np.float32))
                            self.measure_vecs.append(np.zeros(2, np.float32))
                        self.foundBlobs[tmpColorPos] = False
                        self.foundTrajectory[tmpColorPos] = False
                        self.prevticks[tmpColorPos] = 0.
                    if self.counterTrajectories[tmpColorPos] > 25 and not self.foundTrajectory[tmpColorPos]:
                        self.foundTrajectory[tmpColorPos] = True
                    # more than fixed number of points in trajectory, start kf
                    if self.foundTrajectory[tmpColorPos]:
                        # time interval trigger
                        if last_timestamp - self.prevticks[tmpColorPos] > 1:
                            # when over a threshold period
                            precTick = self.prevticks[tmpColorPos]
                            self.prevticks[tmpColorPos] = last_timestamp
                            dT = (self.prevticks[tmpColorPos] - precTick)/1000
                            if self.foundBlobs[tmpColorPos]:
                                self.kfilters[tmpColorPos].transitionMatrix[0, 2] = dT
                                self.kfilters[tmpColorPos].transitionMatrix[1, 3] = dT
                                self.state_vecs[tmpColorPos] = self.kfilters[tmpColorPos].predict()
                            self.measure_vecs[tmpColorPos][0] = clusterCenter[i, 0]*self.w
                            self.measure_vecs[tmpColorPos][1] = clusterCenter[i, 1]*self.h
                            
                            if not self.foundBlobs[tmpColorPos]:
                                # first detection
                                self.kfilters[tmpColorPos].errorCovPre[0, 0] = 1
                                self.kfilters[tmpColorPos].errorCovPre[1, 1] = 1
                                self.kfilters[tmpColorPos].errorCovPre[2, 2] = 2
                                self.kfilters[tmpColorPos].errorCovPre[3, 3] = 2
                                # update measurement [x, y]
                                self.state_vecs[tmpColorPos][0] = self.measure_vecs[tmpColorPos][0]
                                self.state_vecs[tmpColorPos][1] = self.measure_vecs[tmpColorPos][1]
                                self.state_vecs[tmpColorPos][2] = 0
                                self.state_vecs[tmpColorPos][3] = 0
                                self.kfilters[tmpColorPos].statePost = self.state_vecs[tmpColorPos]
                                self.foundBlobs[tmpColorPos] = True
                            else:
                                self.kfilters[tmpColorPos].correct(self.measure_vecs[tmpColorPos])
                                
                    self.counterTrajectories[tmpColorPos] += 1
                    activeTrajectories.append(tmpColorPos)
            prevclusterCenter = clusterCenter
            prev_positionClusterColor = positionClusterColor
            prev_activeTrajectories = activeTrajectories          
            # saving trajectories
            tmp = self.events[start:start+ssize]
            mask = self.activateEvents[start:start+ssize]
            tmp = tmp[mask]
            for i, color in enumerate(positionClusterColor):
                ind = np.where(p2Cluster==i)[0]
                if(len(ind))>0:
                    trajectory = tmp[ind][:, [2, 0, 1]]
                    with open(f'{savedir}/{color}.txt', 'a+') as f:
                        np.savetxt(f, np.c_[trajectory], fmt='%d', delimiter=',') # us, x, y, p
            cnt += 1