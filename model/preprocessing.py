import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing as mp


def connected_components_(img,threshold,i,j,mask):
    
    h,w = img.shape
    stack = [] 
    pos = [[-1,1],[-1,-1],[-1,0],[0,1],[0,-1],[1,1],[1,0],[1,-1]]
    stack.append([i,j])  

    while len(stack) > 0:  
        s = stack[-1]  
        stack.pop() 
        
        i,j = s[0],s[1]
        if mask[i,j] == 0: 
            mask[i,j] = 1 

        for p in pos:
            i,j = p[0]+s[0],p[1]+s[1]
            if i<0 or i==h or j<0 or j==w or mask[i,j]==1 or img[i,j] < threshold:
                continue
            else:
                stack.append([i,j])
    
    return mask


def connected_components(img):

    h,w = img.shape
    mask = np.zeros((h,w)).astype(np.int8)
    idx = np.where(img[20:-20,20:-20] == img[20:-20,20:-20].max())
    threshold = 0.12 * np.max(img.flatten())
    for i,j in zip(*idx):
        mask = connected_components_(img,threshold,i,j,mask)
    
    return mask


def preprocess_util(file):
        img = cv2.imread(file,0)
        last_row = np.where(img[:,0]==255)[0][0]
        img = img[:last_row,:]
        mask = connected_components(img)
        
        img = cv2.imread(file)
        img = img[:last_row,:]
        img = img * np.expand_dims(mask, axis = 2)

        path = os.path.normpath(file)
        path = path.split(os.sep)
        path[-4] = 'real_preprocessed3'
        path[0] = 'C:\\'
        path = os.path.join(*path)
        if not os.path.exists(os.path.split(path)[0]):
            os.makedirs(os.path.split(path)[0])
        cv2.imwrite(path,img)


def main(train_real):

    files = glob.glob(os.path.join(train_real, '*\\images\\*.jpg')) +\
            glob.glob(os.path.join(train_real, '*\\images\\*.png'))
    num_workers = mp.cpu_count()  
    print("Number of processes running in parallel: ", num_workers)
    pool = mp.Pool(processes=num_workers)
   
    for _ in tqdm(pool.imap_unordered(preprocess_util, files), total=len(files)):
        pass
   

if __name__ == "__main__":

    train_real = 'C:\\Users\\sport\\Google Drive\\3 Semester\\Individual Study\\synthetic_foram_model\\data\\real\\'
    main(train_real)