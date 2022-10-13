# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:50:54 2020

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

tScale=20

class ExpCost(BaseCost):

    """Custom cost for exponential signals."""

    # The 2 following attributes must be specified for compatibility.
    model = ""
    min_size = 2

    def fit(self, signal):
        """Set the internal parameter."""
        self.signal = signal
        self.signalN = signal<=8000
        return self

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            float: segment cost
        """
        
        sub = self.signal[start:end]
        #N=np.sum(self.signalN[start:end])+1
        N=end-start#this and run change
        return N*log(np.sum(sub)/N)
        #return (end-start)*log(sub.mean())
        
class QISCost(BaseCost):

    def __init__(self,Nbins):
        self.Nbins = Nbins
    """Custom cost for QIS signals."""

    # The 2 following attributes must be specified for compatibility.
    model = ""
    min_size = 2

    def fit(self, signal):
        """Set the internal parameter."""
        self.signal = signal
        self.signalN = signal<=self.Nbins
        return self

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            float: segment cost
        """
        
        sub = self.signalN[start:end]
        #N=np.sum(self.signalN[start:end])+1
        N=end-start#this and run change
        sumXi = np.sum(sub)
        pHat = 1/N*sumXi
        if pHat>=1:
            pHat=.99
        if pHat <=0:
            pHat=.00001
        return -np.log(pHat)*sumXi-np.log(1-pHat)*(N-sumXi)

class CPWeightExpCost(BaseCost):

    """Custom cost for exponential signals."""

    # The 2 following attributes must be specified for compatibility.
    model = ""
    min_size = 2
    
    def __init__(self,nnClasf,xy,mu,nnAvg):
        self.nnClasf = nnClasf
        self.mu=mu
        self.xy = xy
        self.nnAvg = nnAvg
    def getCP(self):
        return self.changePoints
    
    def fit(self, signal):
        self.signal = signal
        return self

    def error(self, start, end):
        distances, nearestN = self.nnClasf.kneighbors([np.array([self.xy[0],self.xy[1],end])/[1,1,tScale]])
        distances2, nearestN = self.nnClasf.kneighbors([np.array([self.xy[0],self.xy[1],start])/[1,1,tScale]])
        
        extraWeight = (sum(sum(distances))+sum(sum(distances2)))/len(sum(distances))

        sub = self.signal[start:end]
        return (end-start)*log(sub.mean())+self.mu*(extraWeight-.5*self.nnAvg/len(sum(distances)))



    
class Simulate_SinglePixel_Av(object):
    def __init__(self,T,Nbins,pelt_min_size=15,pelt_pen=4,mu=1,cpSolver="pelt"):
        self.T = T
        self.Nbins =Nbins
        self.pelt_min_size=pelt_min_size
        self.pelt_pen=pelt_pen
        self.simulated=False
        self.mu=mu
        self.cpSolver = cpSolver


    def custom_costFunc(self):
        return ExpCost()
    
    def getEdgesPelt(self,phots):
        if self.cpSolver == "window50":
            algo = rpt.Window(width=50,custom_cost=self.custom_costFunc(),min_size=self.pelt_min_size).fit(phots)
        if self.cpSolver == "window100":
            algo = rpt.Window(width=100,custom_cost=self.custom_costFunc(),min_size=self.pelt_min_size).fit(phots)
        elif self.cpSolver == "bottomup":
            algo = rpt.BottomUp(custom_cost=self.custom_costFunc(),min_size=self.pelt_min_size).fit(phots)
        elif self.cpSolver == "binseg":
            algo = rpt.Binseg(custom_cost=self.custom_costFunc(),min_size=self.pelt_min_size).fit(phots)
        else:
            algo = rpt.Pelt(custom_cost=self.custom_costFunc(),min_size=self.pelt_min_size).fit(phots)
        edges = algo.predict(pen=self.pelt_pen)
        edges=np.insert(edges,0,0)
        return edges
    
    def getEdgesCPW(self,phots,NN,xy,nnAvg):
        algo = rpt.Pelt(custom_cost=CPWeightExpCost(NN,xy,self.mu,nnAvg),min_size=self.pelt_min_size).fit(phots)
        edges = algo.predict(pen=self.pelt_pen)
        edges=np.insert(edges,0,0)
        return edges
    
    def gateToPF(self,data):
        sh = np.shape(data)
        phots=[]
        gateLoc =[]
        pt=0
        for k in range(sh[0]):
            t = data[k]
            if t>=self.Nbins+1:
                pt+=self.Nbins
            elif t>0:
                phots.append(pt+t)
                gateLoc.append(k)
                pt=0
        return np.array(phots)*self.T/self.Nbins, np.array(gateLoc)
    
    def getFlux(self,ts):
        t= sum(ts)*self.T/self.Nbins 
        N = sum(ts<self.Nbins)
        return N/t
    
    def initFluxFromChanges(self,measure,cps):
        self.edges=cps
        #does constant values in between edges
        self.simulated_fluxes = []
        for i in range(len(self.edges)-1):
            t = measure[int(self.edges[i]):int(self.edges[i+1])]
            self.simulated_fluxes.append(self.getFlux(t))
    
    def run(self,measure,spaceAv):
        
        if self.simulated:
            return
        
        phots, gateLoc = self.gateToPF(spaceAv)#this and expcost change
        
        phots_edges=self.getEdgesPelt(phots)
        
        self.edges =[0]
        for i in range(1,len(phots_edges)-1):
            self.edges.append(gateLoc[phots_edges[i]])
        self.edges.append(len(measure)-1)
        
        #does constant values in between edges
        self.simulated_fluxes = []
        for i in range(len(self.edges)-1):
            t = measure[int(self.edges[i]):int(self.edges[i+1])]
            self.simulated_fluxes.append(self.getFlux(t))
             
        self.simulated=True
        
    def runCPWeighting(self,measure,NN,xy,nnAvg):        
        phots, gateLoc = self.gateToPF(measure)
        phots_edges=self.getEdgesCPW(phots,NN,xy,nnAvg)
        
        self.edges =[0]
        for i in range(1,len(phots_edges)-1):
            self.edges.append(gateLoc[phots_edges[i]])
        self.edges.append(len(measure)-1)
        
        #does constant values in between edges
        self.simulated_fluxes = []
        for i in range(len(self.edges)-1):
            t = measure[int(self.edges[i]):int(self.edges[i+1])]
            self.simulated_fluxes.append(self.getFlux(t))
             
        self.simulated=True
        
        
    def loadValues(self,fluxes,edges):
        self.simulated=True
        self.simulated_fluxes=fluxes
        self.edges = edges
    
    def getValueAt(self,gateIndex):
        
        whichEdge=np.digitize(gateIndex,self.edges)-1
        if(whichEdge >= len(self.simulated_fluxes)):
            whichEdge=len(self.simulated_fluxes)-1

        return self.simulated_fluxes[whichEdge]
            
    def getEdgesAndFluxes(self):
        return self.simulated_fluxes,self.edges

    def getLengthOfCPSegment(self, index):
        whichEdge=np.digitize(index,self.edges)-1
        if(whichEdge >= len(self.simulated_fluxes)):
            return -1

        return self.edges[whichEdge+1] - self.edges[whichEdge]
    
class Simulate_SinglePixel_Av_QIS(Simulate_SinglePixel_Av):
    def custom_costFunc(self):
        return QISCost(self.Nbins)
    
    def getFlux(self,ts):
        N = sum(ts<self.Nbins)
        pHat = 1/len(ts)*N
        if pHat>=1:
            pHat=.99
        return np.log(1/(1-pHat))/self.T
    
    def run(self,measure,spaceAv):
        
        if self.simulated:
            return
        
        phots, gateLoc = self.gateToPF(spaceAv)#this and expcost change
        
        phots_edges=self.getEdgesPelt(measure)
        
        self.edges =[0]
        for i in range(1,len(phots_edges)-1):
            self.edges.append(phots_edges[i])
        self.edges.append(len(measure)-1)
        
        #does constant values in between edges
        self.simulated_fluxes = []
        for i in range(len(self.edges)-1):
            t = measure[int(self.edges[i]):int(self.edges[i+1])]
            self.simulated_fluxes.append(self.getFlux(t))
             
        self.simulated=True
    
def parallel_block(sPix,ph,combined):
    sPix.run(ph,combined)
    return sPix

def parallel_block_CPW(sPix,ph,NN,xy,nnAvg):
    sPix.runCPWeighting(ph,NN,xy,nnAvg)
    return sPix

class ChangepointVideo(object):
    def __init__(self,photons,T,Nbins,kerSize = (1,1,1),min_size=15,penalty=4,mu=1,cpSolver="pelt",QIS=False):
        self.photons = photons
        self.T = T
        self.Nbins =Nbins
        self.pelt_min_size=min_size
        self.pelt_pen=penalty
        self.simulated=False
        self.kerSize=kerSize
        self.mu=mu
        self.cpSolver = cpSolver
        self.QIS = QIS
    
     
    def run(self):
        if self.simulated:
            return
                
        combined = scipy.signal.convolve(self.photons,np.ones(self.kerSize),'same')
        
        sh = np.shape(self.photons)
        self.singlePixels=[]
        for i in tqdm(range(sh[1])):
            for j in range(sh[2]):
                #print(i)
                #print(j)
                #print("a")
                if self.QIS:
                    self.singlePixels.append(
                   Simulate_SinglePixel_Av_QIS(self.T,self.Nbins,self.pelt_min_size,self.pelt_pen,cpSolver=self.cpSolver))
                else:
                    self.singlePixels.append(
                     Simulate_SinglePixel_Av(self.T,self.Nbins,self.pelt_min_size,self.pelt_pen,cpSolver=self.cpSolver))
                self.singlePixels[-1].run(self.photons[:,i,j],combined[:,i,j])
        self.singlePixels=np.reshape(self.singlePixels,(sh[1],sh[2]))
        self.simulated=True
        
        
    def run_parallel(self,n_cores=4):
        if self.simulated:
            return
        if self.kerSize ==(1,1,1):
            combined = np.copy(self.photons)
        else:
            combined = scipy.signal.convolve(self.photons,np.ones(self.kerSize),'same')
        
        sh = np.shape(self.photons)
        self.singlePixels=[]
        for i in range(sh[1]):
            for j in range(sh[2]):
                if self.QIS:
                    self.singlePixels.append(
                   Simulate_SinglePixel_Av_QIS(self.T,self.Nbins,self.pelt_min_size,self.pelt_pen,cpSolver=self.cpSolver))
                else:
                    self.singlePixels.append(
                     Simulate_SinglePixel_Av(self.T,self.Nbins,self.pelt_min_size,self.pelt_pen,cpSolver=self.cpSolver))
                #self.singlePixels[-1].run(self.photons[:,i,j],combined[:,i,j])
                
        self.singlePixels=np.reshape(self.singlePixels,(sh[1],sh[2]))
        newPixels=[]
        for i in tqdm(range(sh[1])):
            pixLine=Parallel(n_jobs=n_cores)(delayed(parallel_block)(self.singlePixels[i,j],self.photons[:,i,j],combined[:,i,j])for j in range(sh[2]))
            newPixels.append(pixLine)
        self.singlePixels=np.array(newPixels)
        self.simulated=True
    
    def getChangePoints(self):
        size = self.singlePixels.shape
        tMax = self.singlePixels[0][0].edges[-1]
        points = []
        
        for i in range(0,size[0]):
            for j in range(0,size[1]):
                for ed in self.singlePixels[i][j].edges:
                    if ed<tMax and ed>0:
                        points.append([i,j,ed])
        return np.array(points)
    
    def get_NN(self):
        CPS = self.getChangePoints()
        N=20
        nbrs = NearestNeighbors(n_neighbors=N, algorithm='ball_tree').fit(CPS/[1,1,tScale])
        ds,ns=nbrs.kneighbors(CPS/[1,1,tScale])
        return nbrs, np.mean(ds)
    
    def run_parallel_NN_reg(self,n_cores=4): 
        NN,nnAvg=self.get_NN()
        sh = np.shape(self.photons)
        self.singlePixels=[]
        for i in range(sh[1]):
            for j in range(sh[2]):
                if self.QIS:
                    self.singlePixels.append(
                   Simulate_SinglePixel_Av_QIS(self.T,self.Nbins,self.pelt_min_size,self.pelt_pen,cpSolver=self.cpSolver))
                else:
                    self.singlePixels.append(
                     Simulate_SinglePixel_Av(self.T,self.Nbins,self.pelt_min_size,self.pelt_pen,cpSolver=self.cpSolver))
                
        self.singlePixels=np.reshape(self.singlePixels,(sh[1],sh[2]))
        newPixels=[]
        for i in tqdm(range(sh[1])):
            pixLine=Parallel(n_jobs=n_cores)(delayed(parallel_block_CPW)(self.singlePixels[i,j],self.photons[:,i,j],NN,[i,j],nnAvg)for j in range(sh[2]))
            newPixels.append(pixLine)
        self.singlePixels=np.array(newPixels)
        self.simulated=True
    
    def getFluxFrame(self,gateIndex):
        if not self.simulated:
            self.run()
            
        sh = np.shape(self.singlePixels)
        fluxRe = np.zeros(sh)
        for i in range(sh[0]):
            for j in range(sh[1]):
                fluxRe[i,j] = self.singlePixels[i,j].getValueAt(gateIndex)
        return fluxRe 
    
    def getLengthFrame(self,gateIndex):
        if not self.simulated:
            self.run()
            
        sh = np.shape(self.singlePixels)
        fluxRe = np.zeros(sh)
        for i in range(sh[0]):
            for j in range(sh[1]):
                fluxRe[i,j] = self.singlePixels[i,j].getLengthOfCPSegment(gateIndex)
        return fluxRe
    
    def write_h5(self, filename):
        """Write all results to an h5 file
        """
        if not self.simulated:
            self.run()
        
        f = h5py.File(filename, "w")
        dset = f.create_dataset("kerSize", data=self.kerSize)
        dset = f.create_dataset("T", data=self.T)
        #dset = f.create_dataset("photsPerFrame", data=self.photsPerFrame)
        dset = f.create_dataset("Nbins", data=self.Nbins)
        dset = f.create_dataset("pelt_min_size", data=self.pelt_min_size)
        dset = f.create_dataset("pelt_pen", data=self.pelt_pen)
        dset = f.create_dataset("simulated", data=self.simulated)

        sh = np.shape(self.singlePixels)
        dset = f.create_dataset("sh", data=sh)
        for i in range(sh[0]):
            for j in range(sh[1]):
                n = str(i) +","+str(j)
                grp = f.create_group(n)
                fs,es = self.singlePixels[i,j].getEdgesAndFluxes()
                dset2 = grp.create_dataset("fluxes",data=np.array(fs))
                dset2 = grp.create_dataset("edges",data=np.array(es))
    
        f.close()

    def read_h5(self, filename):
        """Read object info from an existing h5 file
        """
        f = h5py.File(filename,'r')
        
        self.kerSize = f['kerSize'][()]
        self.T = f['T'][()]
        #self.photsPerFrame = f['photsPerFrame'][()]
        self.Nbins = f['Nbins'][()]
        self.pelt_min_size= f['pelt_min_size'][()]
        self.pelt_pen= f['pelt_pen'][()]
        self.simulated=f['simulated'][()]

        sh = np.array(f['sh'][()])
        self.singlePixels = []
        
        for i in range(sh[0]):
            for j in range(sh[1]):
                self.singlePixels.append(
                        Simulate_SinglePixel_Av(self.T,self.Nbins,self.pelt_min_size,self.pelt_pen))
                n = str(i) +","+str(j)
                fs = np.array(f[n+"/fluxes"])
                es = np.array(f[n+"/edges"])
                self.singlePixels[-1].loadValues(fs,es)
        self.singlePixels=np.reshape(self.singlePixels,(sh[0],sh[1]))
        print("done loading h5")
        f.close()
        
def normalize(array):
    normalized_array = (array - np.min(array))/(np.max(array) - np.min(array)) 
    img_array = (normalized_array* 255).astype(np.uint8)
    return img_array

def logNormalize(vid):
    return normalize(np.log10(vid+.001))

def cutNorm(vid,percent=.005):
    vid = np.array(vid)
    low=np.percentile(vid.flatten(),(1-percent)*100)
    up=np.percentile(vid.flatten(),percent*100)
    vid[vid>low] =low
    vid[vid<up]=up
    return vid


def loadVideoFromH5(fileName,downSamp=10):
    si = Simulate_Video_Av([1],(1,3,3),0.001/100,100,8000)
    si.read_h5(fileName)
    l=si.singlePixels[0,0].edges[-1]
    toShow=[]
    for i in tqdm(range(0,l,downSamp)):
        toShow.append(si.getFluxFrame(i))
    
   # toShow =normalize(np.array(toShow))
    return toShow,getEds(si)


def getEds(si):
    pix = si.singlePixels
    l=si.singlePixels[0,0].edges[-1]
    es = np.zeros(l+1)
    
    for i in pix:
        for p in i:
           for e in p.edges:
               es[int(e)] +=1
    return es
