from multitrack import MultiTrack
import numpy as np
import pandas as pd
import os

image_size = (180, 240)
def load_dataset(path_dataset, sequence, suffix='txt'):
    if suffix=='txt':
        events = pd.read_csv(
            '{}/{}/events.txt'.format(path_dataset,sequence), sep=" ", header=None)  
        events_set = events.to_numpy()[:, [1, 2, 0, 3]] # [x, y, t, p]
        print("Events total count: ", len(events_set))
        print("Time duration of the sequence: {} s".format(events_set[-1, 2] - events_set[0, 2]))
        events_set[:, 2] *= 1e+6    # s -> us
    events_set = events_set.astype(np.int64)
    events_set[:, 2] -= events_set[0, 2]
    return events_set

if __name__ == '__main__':
    dataset_path, sequence = 'dataset', 'shapes_translation'
    # dataset_path, sequence = 'dataset', 'shapes_rotation'
    # dataset_path, sequence = 'dataset', 'shapes_6dof'
    events_set = load_dataset(dataset_path, sequence, suffix='txt')
    method = MultiTrack(events_set, image_size, max_clusters=256)
    save_dir = './' + sequence + '_tracking_res'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    method.forward(savedir=save_dir)
