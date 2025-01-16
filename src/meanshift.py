import numpy as np

colormap = ['c', 'g', 'r', 'm', 'y', 'b', 'w', 'k']
shapemap = ['v', '1', 's', 'p', '+', '*', 'x', 'o']

def postprocess_cluster(clusterCenter, p2Cluster, threshold):
    """filter the cluster that number of points is less than threshold
    Args:
        clusterCenter (np.array[num_clusters,])
        p2Cluster (np.array[num_pts]): 0 - noise, 1~num_clusters - clusters
        threshold (int)
    """
    clusterCenter_ = np.empty((0, 3), dtype=np.float32)
    cnt = 1
    for cluster_id in range(clusterCenter.shape[0]):
        ind = np.where(p2Cluster==cluster_id)[0]
        if np.count_nonzero(ind) > threshold:
            clusterCenter_ = np.concatenate((clusterCenter_, clusterCenter[cluster_id][None, :]), axis=0)
            p2Cluster[ind] = cnt
            cnt += 1
        else:
            p2Cluster[ind] = 0
    return clusterCenter_, p2Cluster

def meanShiftCluster_Gaussian(pts, bandwidth, threshold=10):
    """Mean Shift Clustering with Gaussian kernel
    Args:
        pts(np.array[num_events, 3], np.float32): count events per pixels with polarity every slide event window
        bandwith(float): radius of searching region
        th(int): the least number of points to be considered as a cluster
    Parameters:
        clusterCenter(np.array[num_clusters,]): (x, y, z) position of clusters center
        clusterVotes(np.array[num_cluster, ]): each of size (num_pts) represents voters of each cluster
    Return:
        clusterCenter(np.array[num_clusters,]): (x, y, z) position of clusters center
        p2Cluster(np.array[num_pts]): the cluster that each point belongs to
    """
    num_pts, _ = pts.shape
    stopThresh = 1e-3*bandwidth   # border condition
    VisitFlag = np.zeros(num_pts, dtype=np.int32)  # is point visited?
    clusterVotes = np.empty((0, num_pts), dtype=np.int32)
    clusterCenter = np.empty((0, 3), dtype=np.float32)
    lambda_ = 10.
    p2Cluster = []
    if num_pts == 0:
        return clusterCenter, np.array(p2Cluster)
    while(VisitFlag.min() == 0):
        # pick random seed pint
        tmpInd = np.random.choice(np.where(VisitFlag==0)[0])
        # initialize mean to every point's location
        Mean = pts[tmpInd]
        thisClusterVote = np.zeros(num_pts, dtype=np.int32)
        
        while(1):
            # calculate distance between every point and 'Mean' center
            pts_cp = pts.copy()
            dis = np.linalg.norm(pts_cp - Mean, 2, 1) # [num_pts, 3] -> [num_pts]
            inlinerInd = dis < bandwidth  # [num_pts]
            VisitFlag[inlinerInd] = 1
            thisClusterVote[inlinerInd] += 1
            if np.count_nonzero(inlinerInd) == 0:
                break
            mask = (inlinerInd==0)
            # Update new center from inliners
            pts_cp[mask] = 0
            old_Mean = Mean.copy()
            Mean = pts_cp.sum(0) / np.count_nonzero(inlinerInd)
            # the weight defined by distance of inliners to center but negative correlation
            # the closer the greater contribution
            weight = np.exp(-(pts_cp - Mean)**2/lambda_) # [num_pts, 3]
            weight[mask] = 0
            # Update center again with weight
            Mean = (pts_cp*weight).sum(0)/(weight.sum(0) + 1e-6)
            mean_shift = np.linalg.norm(Mean-old_Mean, 2, 0)
            if mean_shift < stopThresh:
                if clusterCenter.size != 0:
                    # cluster merging: choose to merge with the closet cluster (Different with usual)
                    clusters_dis = np.linalg.norm(clusterCenter-Mean, 2, 1) # [1, num_cluster]
                    cluster_id = clusters_dis.argmin()
                    if clusters_dis[cluster_id] < stopThresh/2:
                        # Merge
                        clusterCenter[cluster_id] = (clusterCenter[cluster_id] + Mean)*0.5
                        clusterVotes[cluster_id] = thisClusterVote
                        break
                # add new cluster
                clusterCenter = np.concatenate((clusterCenter, Mean[None, :]), axis=0)
                clusterVotes = np.concatenate((clusterVotes, thisClusterVote[None, :]), axis=0)
                # showcluster(pts, Mean, thisClusterVote)
                break
    if clusterCenter.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty([0])
    
    TclusterVote = clusterVotes.transpose((1, 0)) # [num_pts, num_cluster] = [num_cluster, num_pts].T 
    for i in TclusterVote:
        p2Cluster.append(i.argmax())
    # because each point will be assigned with a cluster, we should filter the clusters which number of points is less than a threshold
    # return postprocess_cluster(clusterCenter, np.array(p2Cluster), threshold)
    return clusterCenter, np.array(p2Cluster)
    