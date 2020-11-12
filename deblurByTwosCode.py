# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:39:49 2020

@author: 19522
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 12:23:17 2020

@author: 19522
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:48:58 2020

@author: 19522
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py

#import struct
import os
from joblib import Parallel, delayed
#import multiprocessing
import pickle
import cv2
from tqdm import tqdm
import ruptures as rpt
import scipy
from math import log
from ruptures.base import BaseCost
from sklearn.neighbors import NearestNeighbors
from changepointClasses import * 

def flux_timing(N,t):
    return N/t

def flux_counts(N,t):
    return np.log(1/(1-N/t))

def getWarpMatrix(im1_gray,im2_gray,warp_matrix=None):
    

    # Find size of image1
    
    # Define the motion model
    warp_mode = cv2.MOTION_EUCLIDEAN
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_matrix is None:
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else :
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            
    warp_matrix=np.array(warp_matrix,dtype=np.float32)
    # Specify the number of iterations.
    number_of_iterations =10000;
     
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;
     
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
     
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
    return warp_matrix

#sz = im1_gray.shape
#im2_aligned = cv2.warpAffine(im2_gray, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
# Show final results
    
#cv2.imshow("Image 1", im1_gray)
#cv2.imshow("Image 2", im2_gray)
#cv2.imshow("Aligned Image 2", im2_aligned)
    
def getParamsFromWarp(warp_matrix,r0,r1):
    theta = np.arccos(warp_matrix[0,0])*np.sign(warp_matrix[0,1])
    x0 = warp_matrix[0,-1]+(np.cos(theta)-1)*r0-r1*np.sin(theta)
    x1 = warp_matrix[1,-1]+(np.cos(theta)-1)*r1+r0*np.sin(theta)
    #rotMat = warp_matrix[:2,:2]
    #x0,x1=np.linalg.inv(rotMat)@warp_matrix[:,2]
    return [theta,x0,x1]

def getWarpMatrixFromParams(th,x0,x1,r0,r1):
    return np.array([[np.cos(th),np.sin(th),(1-np.cos(th))*r0+r1*np.sin(th)+x0],[-np.sin(th),np.cos(th),(1-np.cos(th))*r1-r0*np.sin(th)+x1]])
    #return np.array([[np.cos(th),np.sin(th),x0*np.cos(th)+x1*np.sin(th)],[-np.sin(th),np.cos(th),x1*np.cos(th)-x0*np.sin(th)]])

def convertFanData(fanData):
    inN = np.zeros_like(fanData)
    inN[fanData<=8000] =1
    inN[fanData==0]=0
    
    tn=np.zeros_like(fanData)
    
    for i in range(np.shape(tn)[1]):
        for j in range(np.shape(tn)[2]):
            tot=0
            for t  in range(np.shape(tn)[0]):
                tot+=fanData[t,i,j]
                if inN[t,i,j] !=0:
                    tn[t,i,j]=tot
                    tot=0
    return tn, inN

def saveVideo(newVid,name='testVid.mp4'):
    sh = np.shape(newVid[0])
    writer = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'MP4V'), 25, (sh[1], sh[0]), False)
    
    for i in range(len(newVid)):
        writer.write(newVid[i])
    del writer

def sampleCPV(si,photsPer,show=False,scale=15):
    outputVid=[]
    tMax = si.singlePixels[0][0].edges[-1]
    for i in range(0,tMax,photsPer):
        vid = si.getFluxFrame(i)
        outputVid.append(logNormalize(vid))
        if show:
            cv2.imshow("f2",cv2.resize(logNormalize(vid),(np.shape(vid)[1]*scale,scale*np.shape(vid)[0]),interpolation= cv2.INTER_NEAREST))
            if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            
        
    vid = si.getFluxFrame(tMax)
    outputVid.append(logNormalize(vid))
    return outputVid


def normalize0(array):
    array=np.nan_to_num(array)
    return normalize(cutNorm(np.nan_to_num(array)))#logNormalize(cutNorm(array))#




def deblurImages(imgs,photsPerFrame,timeIn,NIn,seg=None,warpInit=None,getWarpedPhotons=False):
    sz = np.shape(timeIn[0])
    sz=(sz[0],sz[1])
    r0=sz[0]/2
    r1=sz[1]/2
    imgL = imgs[0]
    params=[]
    if(len(imgs)==1):
        return NIn[0],timeIn[0],getWarpMatrixFromParams(0,0,0,r0,r1)
    for i in range(1,len(imgs)):
        imgN = imgs[i]
        params.append([(i-1,i),getParamsFromWarp(getWarpMatrix(imgL,imgN,warpInit),r0,r1)])
        imgL = imgN
    params= np.array(params)

    thetas= [p[0] for p in params[:,1]]
    x0s= [p[1] for p in params[:,1]]
    x1s= [p[2] for p in params[:,1]]
    
    th = np.mean(thetas)/photsPerFrame
    x0=np.mean(x0s)/photsPerFrame
    x1 = np.mean(x1s)/photsPerFrame
    out=np.zeros(sz)
    N=np.zeros(sz)
    
    Nmoved=[]
    tMoved =[]
    
    for i in range(int(timeIn.shape[0])):
        warp_matrix=getWarpMatrixFromParams(th*i,x0*i,x1*i,r0,r1)
        
        o=cv2.warpAffine(timeIn[i], warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP);
        oN=cv2.warpAffine(NIn[i], warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP);
        out+=o
        N+=oN
        
        if getWarpedPhotons:
            Nmoved.append(oN)
            tMoved.append(o)
        
        #scale=15
        #vid = imgs[0]
        #cv2.imshow("f1",cv2.resize(vid,(np.shape(vid)[1]*scale,scale*np.shape(vid)[0]),interpolation= cv2.INTER_NEAREST))
        
        #vid = cv2.warpAffine(imgs[1], warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP);
        #cv2.imshow("f2",cv2.resize(vid,(np.shape(vid)[1]*scale,scale*np.shape(vid)[0]),interpolation= cv2.INTER_NEAREST))
        #if cv2.waitKey(10) & 0xFF == ord('q'):
         #   break

    if getWarpedPhotons:
          return N,out,getWarpMatrixFromParams(th*photsPerFrame,x0*photsPerFrame,x1*photsPerFrame,r0,r1), Nmoved,tMoved

    return N,out,getWarpMatrixFromParams(th*photsPerFrame,x0*photsPerFrame,x1*photsPerFrame,r0,r1)


def deblurByTwos(imgs,timeIn,NIn,warpInits=None,flux_func=flux_timing):
    grps=2
    Nouts =[]
    outS = []
    warps=[]
    for i in range(0,len(imgs),grps):
        pin = timeIn[i:(i+grps)]
        nin= NIn[i:(i+grps)]
        
        if warpInits is None or i+1 >=len(warpInits):
            warpInit = None
        else:
            warpInit = warpInits[i+1]
           
        N,out,warpM = deblurImages(imgs[i:i+grps],1,pin,nin,warpInit=warpInit)
        Nouts.append(N)
        outS.append(out)
        warps.append(warpM)
    Nouts =np.array(Nouts)
    outS = np.array(outS)
    vids = normalize0(flux_func(Nouts,outS))
    return vids,outS,Nouts,warps

def deblurByTwosPhotons(imgs,photsPer,timeIn,NIn,getWarpedPhotons=False,flux_func=flux_timing):
    Nouts =[]
    outS = []
    warps=[]
    
    warpedPhotonsN=[np.zeros_like(timeIn[0])]
    warpedPhotonsT=[np.zeros_like(timeIn[0])]
    for i in tqdm(range(0,len(imgs)-1)):
        pin = timeIn[i*photsPer:(i+1)*photsPer]
        nin= NIn[i*photsPer:(i+1)*photsPer]
        if getWarpedPhotons:
            N,out,warpM,wPhotonsN,wPhotonsT = deblurImages(imgs[i:i+2],photsPer,pin,nin,getWarpedPhotons=getWarpedPhotons)
            
            warpedPhotonsN=np.vstack((warpedPhotonsN,wPhotonsN))
            warpedPhotonsT=np.vstack((warpedPhotonsT,wPhotonsT))

        else:
            N,out,warpM = deblurImages(imgs[i:i+2],photsPer,pin,nin,getWarpedPhotons)
        Nouts.append(N)
        outS.append(out)
        warps.append(warpM)
    Nouts =np.array(Nouts)
    outS = np.array(outS)
    vids = normalize0(flux_func(Nouts,outS))
    if getWarpedPhotons:
        return vids,outS,Nouts,warps,warpedPhotonsN[1:],warpedPhotonsT[1:]
    return vids,outS,Nouts,warps

def deblurByTwosPhotons_QIS(imgs,photsPer,NIn,getWarpedPhotons=False):
    Nouts =[]
    warps=[]
    
    warpedPhotonsN=[np.zeros_like(NIn[0])]
    for i in tqdm(range(0,len(imgs)-1)):
        nin= NIn[i*photsPer:(i+1)*photsPer]
        if getWarpedPhotons:
            N,warpM,wPhotonsN,wPhotonsT = deblurImages_QIS(imgs[i:i+2],photsPer,nin,getWarpedPhotons=getWarpedPhotons)
            
            warpedPhotonsN=np.vstack((warpedPhotonsN,wPhotonsN))
        else:
            N,warpM = deblurImages_QIS(imgs[i:i+2],photsPer,nin,getWarpedPhotons)
        Nouts.append(N)
        warps.append(warpM)
    Nouts =np.array(Nouts)
    pHat = 1/photsPer*Nouts
    vids = normalize0(np.log(1/(1-pHat)))
    if getWarpedPhotons:
        return vids,Nouts,warps,warpedPhotonsN[1:]
    return vids,Nouts,warps

                      
                      
def deblurByTwos_QIS(imgs,NIn,photsPer,warpInits=None):
    grps=2
    Nouts =[]
    warps=[]
    for i in range(0,len(imgs),grps):
        nin= NIn[i:(i+grps)]
        
        if warpInits is None or i+1 >=len(warpInits):
            warpInit = None
        else:
            warpInit = warpInits[i+1]
           
        N,warpM = deblurImages_QIS(imgs[i:i+grps],1,nin,warpInit=warpInit)
        Nouts.append(N)
        warps.append(warpM)
    Nouts =np.array(Nouts)
    pHat = 1/photsPer*Nouts
    vids = normalize0(np.log(1/(1-pHat)))
    return vids,Nouts,warps
            
def saveMany(name,vids):
    lenV = len(vids)
    step=1
    if lenV>100:
        step=int(lenV/100)
        
    for i in range(0,len(vids),step):
        scale=15
        vid = vids[i]
        vid=cv2.resize(vid,(np.shape(vid)[1]*scale,scale*np.shape(vid)[0]),interpolation= cv2.INTER_NEAREST)
        cv2.imwrite(name+"_"+str(i).zfill(5)+".png",vid)
    

def deblurByTwosStart(imgs,photsPer,phots,Nin,save=False,file="orangeOutput",n="globalOrange"):
    i=0
    if save:
        name = file+"/"+str(i).zfill(5)
        if not os.path.exists(name):
            os.makedirs(name)
        name = name+"/"+n
        saveMany(name,imgs)
    
    vids, outS,Nouts,warps = deblurByTwosPhotons(imgs,photsPer,phots,Nin)
    vids = np.nan_to_num(vids)
    print(np.shape(warps))
    print(np.shape(vids))

    i=1
    
    if save:
        name = file+"/"+str(i).zfill(5)
        if not os.path.exists(name):
            os.makedirs(name)
        name = name+"/"+n
        saveMany(name,vids)
        
    print("looping")
    
    while(len(vids)>1):
        
        #imgs = imgs[::2]
        #vids,outS,Nouts = deblurByTwos(imgs,outS,Nouts)
        vids,outS,Nouts,warps = deblurByTwos(vids,outS,Nouts,warpInits=warps)
        vids = np.nan_to_num(vids)
        #vids=np.flip(vids,0)
        #outS=np.flip(outS)
        #Nouts = np.flip(Nouts)
        
        i+=1
        
        if save:
            name = file+"/"+str(i).zfill(5)
            if not os.path.exists(name):
                os.makedirs(name)
            name = name+"/"+n
            saveMany(name,vids)  
    return vids, outS,Nouts

def deblurByTwosMovingAvgStart(changePointVid,photsPer,photsInput,step,size,scaleFactor=1,save=False,name="1"):
    photsIn,NumIn=convertFanData(photsInput)
    
    toRe = []
    for i in tqdm(range(0,len(photsIn)-size,step)):
        outputVid=[]
        for j in range(i,i+size,photsPer):
            vid = changePointVid.getFluxFrame(j)
            outputVid.append(logNormalize(vid))
    
        vid = changePointVid.getFluxFrame(i+size)
        outputVid.append(logNormalize(vid))
        
        nextNin = NumIn[i:i+size]
        nextPhots = photsIn[i:i+size]

        outputVid=scipy.ndimage.zoom(outputVid,(1,scaleFactor,scaleFactor) , order=0)      
        nextPhots=scipy.ndimage.zoom(nextPhots,(1,scaleFactor,scaleFactor) , order=0)            
        nextNin=scipy.ndimage.zoom(nextNin,(1,scaleFactor,scaleFactor) , order=0)
        vids, outS,Nouts = deblurByTwosStart(outputVid,photsPer,nextPhots,nextNin,save=False)
        toRe.append(vids[0])
        print(np.shape(vids[0]))
    if save:
        print("saving")
        saveVideo(toRe,name)

    return toRe

def deblurByTwosStartVariableFR(si,cpsPercent,photsInput,save=False,file="orangeOutput",n="globalOrange",flux_func=flux_timing):
    
    tIn,Nin=convertFanData(photsInput)
    sh = np.shape(si.singlePixels)
    cpInc = int(sh[0]*sh[1]*cpsPercent)
    tMax = si.singlePixels[0][0].edges[-1]
    
    cps =si.getChangePoints()
    cpsTimes = cps[cps[:,2].argsort()][:,2]
    cps=[]
    
    
    indeces = [0]
    currCP=0
    done=False
    while not done:
        currCP += cpInc
        if currCP>=len(cpsTimes):
            currCP = len(cpsTimes)-1
        ind = cpsTimes[currCP]
        currCP = np.where(cpsTimes==ind)[0][-1]
        indeces.append(ind)
        if currCP >= len(cpsTimes)-1:
            done=True
    indeces.append(tMax)
        
    
    imgs=[]   
    for i in indeces:
        imgs.append(normalize0(si.getFluxFrame(i)))            
    imgs.append(normalize0(si.getFluxFrame(tMax)))
    print(indeces)
    
    name_ind=0
    if save:
        name = file+"/"+str(name_ind).zfill(5)
        if not os.path.exists(name):
            os.makedirs(name)
        name = name+"/"+n
        saveMany(name,imgs)
    
    vids=[]
    outS=[]
    Nouts=[]
    warps=[]
    photsIndex =0
    for j in range(1,len(indeces)):
        photsPer = indeces[j]-indeces[j-1]
        photsIndex +=photsPer
        vidTemp, outSTemp,NoutsTemp,warpsTemp = deblurByTwosPhotons(imgs[j-1:j+1],photsPer,tIn[photsIndex-photsPer:photsIndex],Nin[photsIndex-photsPer:photsIndex],flux_func=flux_func)
        
        print(np.shape(vidTemp))
        vids.append(vidTemp[0])
        outS.append(outSTemp[0])
        Nouts.append(NoutsTemp[0])
    
    
    vids = np.array(vids)
    outS =np.array(outS)
    Nouts = np.array(Nouts)
    vids = np.nan_to_num(vids)
    print(np.shape(warps))
    print(np.shape(vids))

    name_ind=1
    
    if save:
        name = file+"/"+str(name_ind).zfill(5)
        if not os.path.exists(name):
            os.makedirs(name)
        name = name+"/"+n
        saveMany(name,vids)
        
    print("looping")
    
    while(len(vids)>1):
        
        #imgs = imgs[::2]
        #vids,outS,Nouts = deblurByTwos(imgs,outS,Nouts)
        vids,outS,Nouts,warps = deblurByTwos(vids,outS,Nouts,warpInits=warps,flux_func=flux_func)
        vids = np.nan_to_num(vids)
        #vids=np.flip(vids,0)
        #outS=np.flip(outS)
        #Nouts = np.flip(Nouts)
        
        name_ind+=1
        
        if save:
            name = file+"/"+str(name_ind).zfill(5)
            if not os.path.exists(name):
                os.makedirs(name)
            name = name+"/"+n
            saveMany(name,vids)  
    return vids, outS,Nouts

def parallel_block_phots(imgs,photsPer,phots,Nin,flux_func):
    #sPix.run(ph,combined)
    vids, outS,Nouts,warps =deblurByTwosPhotons(imgs,photsPer,phots,Nin,flux_func=flux_func)
    return vids[0], outS[0],Nouts[0],warps[0]

def parallel_block_no_phots(vids,outS,Nouts,warps,flux_func):
    #sPix.run(ph,combined)        
    vids, outS,Nouts,warps =deblurByTwos(vids,outS,Nouts,warpInits=warps,flux_func=flux_func)
    return vids[0], outS[0],Nouts[0],warps[0]
                      
def deblurByTwosStartVariableFRMulti(si,cpsPercent,photsInput,NPercent=.05,n_cores=4,save=False,file="orangeOutput",n="globalOrange",flux_func=flux_timing):
    
        
    timeIn,NIn=convertFanData(photsInput)
    if flux_func==flux_counts:
        timeIn = np.ones(np.shape(NIn))
    needN = np.sum(NIn)*NPercent

    sh = np.shape(si.singlePixels)
    cpInc = int(sh[0]*sh[1]*cpsPercent)
    tMax = si.singlePixels[0][0].edges[-1]

    cps =si.getChangePoints()
    cpsTimes = cps[cps[:,2].argsort()][:,2]
    cps=[]
    indeces = [0]
    currCP=0
    done=False
    while not done:
        currCP += cpInc
        if currCP>=len(cpsTimes):
            currCP = len(cpsTimes)-1
        ind = cpsTimes[currCP]
        nSum =np.sum(NIn[indeces[-1]:ind])
        currCP = np.where(cpsTimes==ind)[0][-1]
        if(nSum>=needN):
            indeces.append(ind)
        if currCP >= len(cpsTimes)-1:
            done=True
    indeces.append(tMax)

    
    imgs=[]   
    for i in indeces:
        imgs.append(normalize0(si.getFluxFrame(i)))            
    #imgs.append(normalize0(si.getFluxFrame(tMax)))
    print(indeces)
    
    name_ind=0
    if save:
        name = file+"/"+str(name_ind).zfill(5)
        if not os.path.exists(name):
            os.makedirs(name)
        name = name+"/"+n
        saveMany(name,imgs)
    
    temp=Parallel(n_jobs=n_cores)(delayed(parallel_block_phots)(imgs[i:i+2],indeces[i+1]-indeces[i],timeIn[indeces[i]:indeces[i+1]],NIn[indeces[i]:indeces[i+1]],flux_func)for i in range(0,len(imgs)-1))
    
    temp=np.array(temp)
    vids=[]
    for i in temp[:,0]:
        vids.append(i)
    vids=np.array(vids)
    outS=[]
    for i in temp[:,1]:
        outS.append(i)
    outS=np.array(outS)
    Nouts=[]
    for i in temp[:,2]:
        Nouts.append(i)
    Nouts=np.array(Nouts)
    warps=[]
    for i in temp[:,3]:
        warps.append(i)
    warps=np.array(warps)
    
    vids = np.nan_to_num(vids)
        #vids=np.flip(vids,0)
        #outS=np.flip(outS)
        #Nouts = np.flip(Nouts)
    print(np.shape(vids))
   
    
    name_ind=1
    
    if save:
        name = file+"/"+str(name_ind).zfill(5)
        if not os.path.exists(name):
            os.makedirs(name)
        name = name+"/"+n
        saveMany(name,vids)
        
    print("looping")
    
    while(len(vids)>1):
        #imgs = imgs[::2]
        #vids,outS,Nouts = deblurByTwos(imgs,outS,Nouts)
        temp=Parallel(n_jobs=n_cores)(delayed(parallel_block_no_phots)(vids[i:i+2],outS[i:i+2],Nouts[i:i+2],warps[i:i+2],flux_func)for i in range(0,len(vids)-1,2))
        temp=np.array(temp)
        vids=[]
        for i in temp[:,0]:
            vids.append(i)
        vids=np.array(vids)
        outS=[]
        for i in temp[:,1]:
            outS.append(i)
        outS=np.array(outS)
        Nouts=[]
        for i in temp[:,2]:
            Nouts.append(i)
        Nouts=np.array(Nouts)
        warps=[]
        for i in temp[:,3]:
            warps.append(i)
        warps=np.array(warps)
        
        vids = np.nan_to_num(vids)
        #vids=np.flip(vids,0)
        #outS=np.flip(outS)
        #Nouts = np.flip(Nouts)
        print(np.shape(vids))
        
        name_ind+=1
        
        if save:
            name = file+"/"+str(name_ind).zfill(5)
            if not os.path.exists(name):
                os.makedirs(name)
            name = name+"/"+n
            saveMany(name,vids)  
    return vids, outS,Nouts


def deblurByTwosStartMulti(imgs,photsPer,timeIn,NIn,n_cores=4,save=False,file="orangeOutput",n="globalOrange",flux_func=flux_timing):
    i=0
    if save:
        name = file+"/"+str(i).zfill(5)
        if not os.path.exists(name):
            os.makedirs(name)
        name = name+"/"+n
        saveMany(name,imgs)
    
    temp=Parallel(n_jobs=n_cores)(delayed(parallel_block_phots)(imgs[i:i+2],photsPer,timeIn[i*photsPer:(i+1)*photsPer],NIn[i*photsPer:(i+1)*photsPer],flux_func)for i in range(0,len(imgs)-1))
    temp=np.array(temp)
    vids=[]
    for i in temp[:,0]:
        vids.append(i)
    vids=np.array(vids)
    outS=[]
    for i in temp[:,1]:
        outS.append(i)
    outS=np.array(outS)
    Nouts=[]
    for i in temp[:,2]:
        Nouts.append(i)
    Nouts=np.array(Nouts)
    warps=[]
    for i in temp[:,3]:
        warps.append(i)
    warps=np.array(warps)
        
    vids = np.nan_to_num(vids)
    print(np.shape(warps))
    print(np.shape(vids))

    i=1
    
    if save:
        name = file+"/"+str(i).zfill(5)
        if not os.path.exists(name):
            os.makedirs(name)
        name = name+"/"+n
        saveMany(name,vids)
        
    print("looping")
    
    while(len(vids)>1):
        
        #imgs = imgs[::2]
        #vids,outS,Nouts = deblurByTwos(imgs,outS,Nouts)
        temp=Parallel(n_jobs=n_cores)(delayed(parallel_block_no_phots)(vids[i:i+2],outS[i:i+2],Nouts[i:i+2],warps[i:i+2],flux_func)for i in range(0,len(vids)-1,2))
        temp=np.array(temp)
        vids=[]
        for i in temp[:,0]:
            vids.append(i)
        vids=np.array(vids)
        outS=[]
        for i in temp[:,1]:
            outS.append(i)
        outS=np.array(outS)
        Nouts=[]
        for i in temp[:,2]:
            Nouts.append(i)
        Nouts=np.array(Nouts)
        warps=[]
        for i in temp[:,3]:
            warps.append(i)
        warps=np.array(warps)
        
        vids = np.nan_to_num(vids)
        #vids=np.flip(vids,0)
        #outS=np.flip(outS)
        #Nouts = np.flip(Nouts)
        print(np.shape(vids))
        
        i+=1
        
        if save:
            name = file+"/"+str(i).zfill(5)
            if not os.path.exists(name):
                os.makedirs(name)
            name = name+"/"+n
            saveMany(name,vids)  
    return vids, outS,Nouts

def parallel_block_phots_QIS(imgs,photsPer,Nin):
    #sPix.run(ph,combined)
    vids,Nouts,warps =deblurByTwosPhotons_QIS(imgs,photsPer,Nin)
    return vids[0],Nouts[0],warps[0],photsPer

def parallel_block_no_phots_QIS(vids,Nouts,photsPer,warps):
    #sPix.run(ph,combined)        
    vids,Nouts,warps =deblurByTwos_QIS(vids,Nouts,photsPer,warpInits=warps)
    return vids[0],Nouts[0],warps[0],photsPer
                      
def deblurByTwosStartVariableFRMulti_QIS(si,cpsPercent,photsInput,NPercent=.05,n_cores=4,save=False,file="orangeOutput",n="globalOrange"):
    i=0
    if save:
        name = file+"/"+str(i).zfill(5)
        if not os.path.exists(name):
            os.makedirs(name)
        name = name+"/"+n
        saveMany(name,imgs)
        
    timeIn,NIn=convertFanData(photsInput)
    
    needN = np.sum(NIn)*NPercent

    sh = np.shape(si.singlePixels)
    cpInc = int(sh[0]*sh[1]*cpsPercent)
    tMax = si.singlePixels[0][0].edges[-1]

    cps =si.getChangePoints()
    cpsTimes = cps[cps[:,2].argsort()][:,2]
    cps=[]
    indeces = [0]
    currCP=0
    done=False
    while not done:
        currCP += cpInc
        if currCP>=len(cpsTimes):
            currCP = len(cpsTimes)-1
        ind = cpsTimes[currCP]
        nSum =np.sum(NIn[indeces[-1]:ind])
        currCP = np.where(cpsTimes==ind)[0][-1]
        if(nSum>=needN):
            indeces.append(ind)
        if currCP >= len(cpsTimes)-1:
            done=True
    indeces.append(tMax)

    
    imgs=[]   
    for i in indeces:
        imgs.append(normalize0(si.getFluxFrame(i)))            
    #imgs.append(normalize0(si.getFluxFrame(tMax)))
    print(indeces)

    
    temp=Parallel(n_jobs=n_cores)(delayed(parallel_block_phots_QIS)(imgs[i:i+2],indeces[i+1]-indeces[i],NIn[indeces[i]:indeces[i+1]])for i in range(0,len(imgs)-1))
    
    temp=np.array(temp)
    vids=[]
    for i in temp[:,0]:
        vids.append(i)
    vids=np.array(vids)
    Nouts=[]
    for i in temp[:,1]:
        Nouts.append(i)
    Nouts=np.array(Nouts)
    warps=[]
    for i in temp[:,2]:
        warps.append(i)
    warps=np.array(warps)
    photsPer=[]
    for i in temp[:,3]:
        photsPer.append(i)
    
    vids = np.nan_to_num(vids)
        #vids=np.flip(vids,0)
        #outS=np.flip(outS)
        #Nouts = np.flip(Nouts)
    print(np.shape(vids))
   

    i=1
    
    if save:
        name = file+"/"+str(i).zfill(5)
        if not os.path.exists(name):
            os.makedirs(name)
        name = name+"/"+n
        saveMany(name,vids)
        
    print("looping no")
    
    while(len(vids)>1):
        #imgs = imgs[::2]
        #vids,outS,Nouts = deblurByTwos(imgs,outS,Nouts)
        temp=Parallel(n_jobs=n_cores)(delayed(parallel_block_no_phots_QIS)(vids[i:i+2],Nouts[i:i+2],sum(photsPer[i:i+2]),warps[i:i+2])for i in range(0,len(vids)-1,2))
        temp=np.array(temp)
        vids=[]
        for i in temp[:,0]:
            vids.append(i)
        vids=np.array(vids)
        Nouts=[]
        for i in temp[:,1]:
            Nouts.append(i)
        Nouts=np.array(Nouts)
        warps=[]
        for i in temp[:,2]:
            warps.append(i)
        warps=np.array(warps)
        photsPer=[]
        for i in temp[:,3]:
            photsPer.append(i)
        
        vids = np.nan_to_num(vids)
        #vids=np.flip(vids,0)
        #outS=np.flip(outS)
        #Nouts = np.flip(Nouts)
        print(np.shape(vids))
        
        i+=1
        
        if save:
            name = file+"/"+str(i).zfill(5)
            if not os.path.exists(name):
                os.makedirs(name)
            name = name+"/"+n
            saveMany(name,vids)  
    return vids,Nouts, photsPer
                      
def deblurByTwosStartMulti_QIS(imgs,photsPer,timeIn,NIn,n_cores=4,save=False,file="orangeOutput",n="globalOrange"):
    i=0
    if save:
        name = file+"/"+str(i).zfill(5)
        if not os.path.exists(name):
            os.makedirs(name)
        name = name+"/"+n
        saveMany(name,imgs)
    
    temp=Parallel(n_jobs=n_cores)(delayed(parallel_block_phots_QIS)(imgs[i:i+2],photsPer,NIn[i*photsPer:(i+1)*photsPer])for i in range(0,len(imgs)-1))
    temp=np.array(temp)
    vids=[]
    for i in temp[:,0]:
        vids.append(i)
    vids=np.array(vids)
    Nouts=[]
    for i in temp[:,1]:
        Nouts.append(i)
    Nouts=np.array(Nouts)
    warps=[]
    for i in temp[:,2]:
        warps.append(i)
    warps=np.array(warps)
    photsPer=[]
    for i in temp[:,3]:
        photsPer.append(i)
    vids = np.nan_to_num(vids)
    print(np.shape(warps))
    print(np.shape(vids))

    i=1
    
    if save:
        name = file+"/"+str(i).zfill(5)
        if not os.path.exists(name):
            os.makedirs(name)
        name = name+"/"+n
        saveMany(name,vids)
        
    print("looping")
    
    while(len(vids)>1):
        
        #imgs = imgs[::2]
        #vids,outS,Nouts = deblurByTwos(imgs,outS,Nouts)
        temp=Parallel(n_jobs=n_cores)(delayed(parallel_block_no_phots_QIS)(vids[i:i+2],Nouts[i:i+2],sum(photsPer[i:i+2]),warps[i:i+2])for i in range(0,len(vids)-1,2))
        temp=np.array(temp)
        vids=[]
        for i in temp[:,0]:
            vids.append(i)
        vids=np.array(vids)

        Nouts=[]
        for i in temp[:,1]:
            Nouts.append(i)
        Nouts=np.array(Nouts)
        warps=[]
        for i in temp[:,2]:
            warps.append(i)
        warps=np.array(warps)
        photsPer=[]
        for i in temp[:,3]:
            photsPer.append(i)
            vids = np.nan_to_num(vids)
        #vids=np.flip(vids,0)
        #outS=np.flip(outS)
        #Nouts = np.flip(Nouts)
        print(np.shape(vids))
        
        i+=1
        
        if save:
            name = file+"/"+str(i).zfill(5)
            if not os.path.exists(name):
                os.makedirs(name)
            name = name+"/"+n
            saveMany(name,vids)  
    return vids,Nouts, photsPer