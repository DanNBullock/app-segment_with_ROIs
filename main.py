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
    operationCheckList=[["any"], ["all"],bothList,eitherList]
    #now run across them and check against what was entered
    validOperationLocation=[currentRequestedOperation in iLists for iLists in operationCheckList]
    if np.any(validOperationLocation):
        operationsVec[iRequests]=validOperationList[np.where(validOperationLocation)[0]]
    else:
        raise ValueError('Input operation specification for request ' + str(iRequests) +' could not be mapped to an avaialble ROI.')




tractogramIn=nib.streamlines.load(config['tractogram'])
streamlines=tractogramIn.streamlines



boolOut=segmentTractMultiROI_fast(streamlines, roisVec, includeVec, operationsVec)