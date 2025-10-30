from multitrack import MultiTrack
import numpy as np
import pandas as pd
import os

image_size = (180, 240)
def load_dataset(path_dataset, sequence, fname='', suffix='txt'):
    if sequence=='star_tracking': # 1ms
        events = pd.read_csv(
            '{}/{}/{}'.format(path_dataset, sequence, fname), sep=",", header=None)  
        events.columns = ['timestamp', 'y', 'x', 'polarity'] 
        events_set = events.to_numpy()
        events_set = events_set[:, [2, 1, 0, 3]] # [t, y, x, p] -> [x, y, t, p]
        take_id = np.logical_and(np.logical_and(np.logical_and(events_set[:, 0] >= 0, \
                                                               events_set[:, 1] >= 0), \
                                                               events_set[:, 0] < 240), \
                                                               events_set[:, 1] < 180)
        events_set = events_set[take_id]
        print("Time duration of the sequence: {} s".format(events_set[-1, 2]*1e-3))
        events_set[:, 2] *= 1e+3 # us
        print("Events total count: ", len(events_set))
    elif suffix=='txt':
        events = pd.read_csv(
            '{}/{}/events.txt'.format(path_dataset,sequence), sep=" ", header=None)  
        events_set = events.to_numpy()[:, [1, 2, 0, 3]] # [x, y, t, p]
        print("Events total count: ", len(events_set))
        print("Time duration of the sequence: {} s".format(events_set[-1, 2] - events_set[0, 2]))
        events_set[:, 2] *= 1e+6    # s -> us
    else:
        events = pd.read_csv(
            '{}/{}/{}'.format(path_dataset, sequence, fname), sep=",", header=None)  
        events.columns = ['x', 'y', 'p', 't']  
        events_set = events.to_numpy()[:, [0, 1, 3, 2]]
        events_set = events_set[np.argsort(events_set[:, 2])]
        print("Events total count: ", len(events_set))
        print("Time duration of the sequence: {} s".format((events_set[-1, 2] - events_set[0, 2])*1e-6))
    events_set = events_set.astype(np.int64)
    events_set[:, 2] -= events_set[0, 2]
    return events_set

if __name__ == '__main__':
    '''ECD'''    
    dataset_path, sequence, fname = 'dataset', 'shapes_translation', ''
    # dataset_path, sequence, fname = 'dataset', 'shapes_rotation', ''
    # dataset_path, sequence, fname = 'dataset', 'shapes_6dof', ''
    events_set = load_dataset(dataset_path+'/', sequence, fname, suffix='txt')
    method = MultiTrack(events_set, image_size, max_clusters=256, bandwidth = 0.15)
    save_dir = './' + '/' + sequence + fname[:-4] + '_tracking_res'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    method.forward(savedir=save_dir)

    
    '''star tracking'''
    # dataset_path, sequence, fname, image_size = 'dataset', 'star_tracking', 'Sequence1.csv', (180, 240) # us
    # events_set = load_dataset(dataset_path, sequence, fname, suffix='csv')
    # method = MultiTrack(events_set, image_size, max_clusters=256, bandwidth = 0.07)
    # save_dir = './' + sequence + '/' +fname[:-4]
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # method.forward(savedir=save_dir)
    
    '''SOD'''
    # dataset_path, sequence, image_size, dt = 'dataset', 'small_objects/sphere-20', (720, 1280), 10000 # dt
    # dataset_path, sequence, image_size, dt = 'dataset', 'small_objects/sphere-30', (720, 1280), 10000 # dt
    # dataset_path, sequence, image_size, dt = 'dataset', 'small_objects/waterdrops', (720, 1280), 10000 # dt
    # files = os.listdir(f'./{dataset_path}/{sequence}')
    # for id, fname in enumerate(files):
    #     # print(dataset_path+'/'+mid, sequence, fname)
    #     # exit()
    #     events_set = load_dataset(dataset_path, sequence, fname, suffix='csv')
    #     method = MultiTrack(events_set, image_size, max_clusters=256, bandwidth = 0.03)
    #     save_dir = './' + sequence + '/' +fname[:-4]
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     method.forward(savedir=save_dir)

    