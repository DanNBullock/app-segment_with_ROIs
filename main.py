#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:51:25 2022

@author: dan
"""

import nibabel as nib
import os
import sys

sys.path.append('./wma_pyTools')
startDir=os.getcwd()
#some how set a path to wma pyTools repo directory
#wmaToolsDir='../wma_pyTools'
#wmaToolsDir='..'
import os
#os.chdir(wmaToolsDir)
print(os.getcwd())
print(os.listdir())
import wmaPyTools.roiTools
import wmaPyTools.analysisTools
import wmaPyTools.segmentationTools
import wmaPyTools.streamlineTools
import wmaPyTools.visTools
import numpy as np
import dipy.io.streamline
import dipy.tracking.utils as ut
#os.chdir(startDir)

import re
import subprocess
import os
import json
import pandas as pd
import nibabel as nib
from nilearn.image import resample_to_img
from glob import glob

# load inputs from config.json
with open('config.json') as config_json:
	config = json.load(config_json)

tractName=config['tractName']

availableROIs=os.path.join(config['availableROIs'],'*.nii.gz')

segRequests=config['segRequests']
#example requests on single lines:

#'roiName any true'
#'roiName either_endpoint true'
splitRequests=segRequests.splitlines()
roisVec=[]
includeVec=[]
operationsVec=[]
for iRequests in splitRequests:
    #clean leading and lagging whitespace
    currentRequest=iRequests.strip()
    #find the space locations
    spaceLocations=[i.start() for i in re.finditer(' ', currentRequest)]
    #first entry is proably the name
    roisVec.append(currentRequest[0:spaceLocations[0]])
    #last entry is the keep/exclude specificiation
    #better not have a space at the end...?
    includeVec.append(currentRequest[spaceLocations[-1]:])
    #whatever is in the middle I guess is the operation specification
    operationsVec=currentRequest[spaceLocations[0]:spaceLocations[-1]].strip()

#get just the names
justROInames=[os.path.basename(iROI) for iROI in availableROIs]
#now clean, modify, and validate the request entry components
for iRequests in range(len(splitRequests)):
    #ROI
    #determine if the ROI is available
    #if they just entered a number...
    if roisVec[iRequests].isnumeric():
        #see if it *uniquely* corresponds to an avaialble ROI
        roiBoolVec=[roisVec[iRequests] in roiName for roiName in justROInames]
        #if there is a single match
        if len(np.where(roiBoolVec)[0])==1:
            #change the name to the returned ROI
            roisVec[iRequests]=availableROIs[np.where(roiBoolVec)[0]]
        else:
            raise ValueError('Input numeric ROI specification for request ' + str(iRequests) +' could not be mapped to an avaialble ROI.')
    #otherwise, I guess it's a string of some sort        
    else:
        #try and find a match
        matchBool=[roisVec[iRequests].lower()==iROI.lower() for iROI in justROInames]
        if len(np.where(matchBool)[0])==1:
            #change the name to the returned ROI
            roisVec[iRequests]=availableROIs[np.where(matchBool)[0]]
        else:
            raise ValueError('Input ROI specification for request ' + str(iRequests) +' could not be mapped to an avaialble ROI.')

    #INCLUDE
    #super overly broad inference as to what constitutes a "keep"-type operation
    #everything else is assumed to be a not
    includeVec[iRequests]=includeVec[iRequests].lower() in ['true', '1', 'keep', 'and', 'include', 'yes']
    
    #OPERATIONS
    currentRequestedOperation=operationsVec[iRequests]
    #theoretically, these are the only ones the program will accept:
    # >  "any" : any point is within tol from ROI. The default.
    # >  "all" : all points are within tol from ROI.
    # >  "either_end" : either of the end-points is within tol from ROI
    # >  "both_end" : both end points are within tol from ROI.
    validOperationList=["any","all","either_end","both_end"]
    #any and all are pretty straightforward, but either_end and both_end have a lot of potential variants
    bothList=['both_end','both end','both_ends','both ends','both','bothends','bothend','endpoints']
    #actually kind of ambiguous with "endpoint" because may be they want only one, which isn't currently implemented
    eitherList=['either_end','either end','either_ends','either ends','either','eitherends','eitherend','endpoint']
    #now create a list to check against
    #MAKE SURE THESE ARE IN THE RIGHT ORDER
    operationCheckList=[["any"], ["all"],eitherList,bothList]
    #now run across them and check against what was entered
    validOperationLocation=[currentRequestedOperation in iLists for iLists in operationCheckList]
    if np.any(validOperationLocation):
        operationsVec[iRequests]=validOperationList[np.where(validOperationLocation)[0]]
    else:
        raise ValueError('Input operation specification for request ' + str(iRequests) +' could not be mapped to an avaialble ROI.')

#just for fun we can put these in a pandas dataframe, print its contents to the console
#and save it down as a csv
segCommandsDF=pd.Dataframe(columns=['roisVec','operationsVec','includeVec'])
segCommandsDF['roisVec']=roisVec
segCommandsDF['operationsVec']=operationsVec
segCommandsDF['includeVec']=includeVec

#load the tractogram
tractogramIn=nib.streamlines.load(config['tractogram'])
streamlines=tractogramIn.streamlines

#laod the rois
roisVecIn=[nib.load(iROI) for iROI in roisVec]

#I don't think they are passed as bool masks, but rather as float
#all the same, a simple sum should work
roiVolSums=[np.sum(iROI.get_fdata()) for iROI in roisVecIn]
#now add that as a column
segCommandsDF['roiVoxelTotal']=roiVolSums

#take this opportunity to "zhoosh" up the name
if tractName.lower() in [None, 'none', 'default', '']:
    print('No unique/informative name entered.\nMerging criteria to form name')
    #reobtain the roiNames (these may have changed via the earlier matching process)
    roiNames=[os.path.basename(iROI) for iROI in roisVec]
    #translate the includeVec
    includeMeanings=['AND' if iInclude  else 'NOT' for iInclude in includeVec]          
    #zip it up into a list of lists
    zippedSegCommands=zip(includeMeanings,roiNames,operationsVec)
    #now for the big join, -- between criteria, _ between operation components
    tractName='--'.join(['_'.join(iCriterion for iCriterion in zippedSegCommands)])          
    

print('Begnning *fast* segmentation of '+ str(len(streamlines)) +' to obtain \n'+ tractName+ '\nwith the following commands:' )
print(segCommandsDF)

#do the fast segmentation
#all kinds of tricks in there to make it go faster than standard dipy segmentation
#pretty sure it is equivalent, but not 100%, previous testing suggested it was,
#and also that the precompute phases led to a speedup floor (i.e. min seg time)
#that was ~ 2 min for ~ 1 mil .5 mm sampled streamlines
boolOut=wmaPyTools.roiTools.segmentTractMultiROI_fast(streamlines, roisVecIn, includeVec, operationsVec)

#turn the boolvec into a wmc
outWmc=wmaPyTools.streamlineTools.updateClassification(boolOut,tractName,existingClassification=None)

outDir='output'
if not os.path.exists(outDir):
    os.makedirs(outDir)

from scipy.io import savemat
#save down the classification structure
savemat(os.path.join(outDir,'wmc','classification.mat'),outWmc)
#hold off on saving the tck down for now
#it's unclear what can or should be done about the naming conventions.
#consider maybe a tcks https://brainlife.io/datatype/5dcf0047c4ae28d7f2298f48
#wmaPyTools.streamlineTools.stubbornSaveTractogram(streamlines,os.path.join(outDir,tractName+'.tck'))