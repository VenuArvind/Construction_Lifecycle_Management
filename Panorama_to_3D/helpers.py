from scipy.ndimage import convolve 
import numpy as np 
import cv2 
import scipy 

def find_n_peaks(signal,prominence,distance,N):
    locs, _ = scipy.signal.find_peaks(signal,
                                      prominence=prominence,
                                      distance=distance)
    pks = signal[locs]
    pk_id = np.argsort(-pks)
    pk_loc = locs[pk_id[:min(N, len(pks))]]
    pk_loc = np.sort(pk_loc)
    return pk_loc, signal[pk_loc]


def get_initial_corners(cor_img,d1=21,d2=3):
    print("Predicting corners")
    cor_img = cor_img[0][0]
    #applying convolution 
    signal = convolve(cor_img, np.ones((d1,d1)),mode="constant", cval=0.0)

    X_loc = find_n_peaks(signal.sum(0), prominence=None,
                         distance=20, N=4)[0]
    cor_id = []

    for x in X_loc:
        x_ = int(np.round(x))

        V_signal = signal[:, max(0, x_-d2):x_+d2+1].sum(1)
        y1, y2 = find_n_peaks(V_signal, prominence=None,
                              distance=20, N=2)[0]
        cor_id.append((x, y1))
        cor_id.append((x, y2))
    #finding peak 
    cor_id = np.array(cor_id) 
    
    return cor_id 


    
