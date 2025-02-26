import numpy as np
import scipy.optimize as spo
from sklearn.linear_model import LinearRegression
from skimage.filters import threshold_otsu
import step_detect
import sklearn.cluster as skc
import skimage
from scipy.ndimage.filters import gaussian_filter1d


def locateBands(region, sensitivity = 0.3, mode = "average", channel = 1):
    #Integrate the profile
    
    averageSlice = np.average(region,axis=1)

    data1D = []
    
    if mode == "average":
        data1D = np.average(averageSlice, axis =1 )
    elif mode == "channel":
        data1D = averageSlice[i][:,channel]

    dataNormed = data1D - np.min(data1D)
    dataNormed = dataNormed / np.max(dataNormed)

    gauss_filtered = gaussian_filter1d(dataNormed, 3, order=1)
    gauss_filtered = gauss_filtered / np.max(gauss_filtered)

    positiveRegion = np.clip(gauss_filtered, 0, 1.0)
    negativeRegion = np.clip(gauss_filtered, -1.0, 0.0)
    
    steps = step_detect.find_steps(positiveRegion, sensitivity)
    
    return positiveRegion, steps #currently just the top edge, shold look at the (top-bot)/2 or (top + width/2)

def locateBandsAll(region, sensitivity = 0.3, mode = "average", channel = 1):
    #Integrate the profile
    
    averageSlice = np.average(region,axis=1)

    data1D = []
    
    if mode == "average":
        data1D = np.average(averageSlice, axis =1 )
    elif mode == "channel":
        data1D = averageSlice[i][:,channel]

    dataNormed = data1D - np.min(data1D)
    dataNormed = dataNormed / np.max(dataNormed)

    gauss_filtered = gaussian_filter1d(dataNormed, 3, order=1)
    gauss_filtered = gauss_filtered / np.max(gauss_filtered)

    positiveRegion = np.clip(gauss_filtered, 0, 1.0)
    negativeRegion = np.clip(gauss_filtered, -1.0, 0.0)
    
    steps = step_detect.find_steps(positiveRegion, sensitivity)
    nsteps = step_detect.find_steps(negativeRegion, sensitivity)
    
    return positiveRegion, negativeRegion, steps, nsteps #currently just the top edge, shold look at the (top-bot)/2 or (top + width/2)

# y = c2 * x^(c1) + c3
def powerfunc(x, pow, coef, C):
    inter = np.power(x, pow)
    return np.multiply(inter, coef)+C
    
def powerfit(fitdataY, fitdataX = []):
    fitdataY = fitdataY 
    
    if len(fitdataX) == 0:
        fitdataX = np.arange(len(fitdataY))
    
    inits = [0.9, 1, min(fitdataY)]
    lowerbounds = [0.001, 0.001, 0]
    upperbounds = [100, 2.5, 1000]

    fitresult = spo.curve_fit(powerfunc, fitdataX, fitdataY, p0 = inits, bounds = [lowerbounds, upperbounds])

    return fitresult,fitdataX,fitdataY 

def spacingsFromSteps(steps):
    spacings = []
    for i in range(1, len(steps)):
        spacings.append(steps[i] - steps[i-1])

    return spacings

def ratiosFromSpacings(spacings):
    ratios = []
    for i in range(0, len(spacings)-1):
        ratios.append(spacings[i+1]/spacings[i])

    return ratios
            
#We sometimes want to reject the final step because it is the spuriouslly detected solvent front and/or only partially formed thickness
#We often want to reject noise near the top of the vial near the interface
def preprocessSteps(steps, rejection = 1.15, rejectLast = True):
    if rejectLast:
        secondToLast = steps[-2] - steps[-3]
        lastSpacing = steps[-1] - steps[-2]
        if lastSpacing/secondToLast > rejection:
            steps = steps[:-1]

    spacings = spacingsFromSteps(steps)
    
    #Look through it backwards for a gap:
    #This does a good job of rejecting the early bumps/noise
    gap = None
    for i in range(len(spacings)-2, -1, -1):
        if gap == None:
            ratio = spacings[i]/spacings[i+1]
            if ratio > rejection:
                gap = i

    if gap != None:
        steps = steps[gap+1:]

    #Also look forward to catch gaps near the back (usually where signal looks weak)
    spacings = spacingsFromSteps(steps) #recalculate here
    
    gap = None
    for i in range(0, len(spacings)-1, 1):
        if gap == None:
            ratio = spacings[i+1]/spacings[i]
            if ratio > 2.0:
                gap = i
                
    if gap != None:
        #print("Forward gap")
        steps = steps[:gap+2]
    
    return steps

#returns processed steplist when given just the image region
def proccessRegion(region, sensitivity = 0.25, rejection = 1.15, rejectLast = True):
    pos, steps = locateBands(region, sensitivity)

    procSteps = preprocessSteps(steps)

    finalSpacings = spacingsFromSteps(procSteps)
    
    res, X, Y = powerfit(finalSpacings)

    stepData = [pos, steps, procSteps]
    fitData = res[0]

    return stepData, fitData



def filterDoubles(rawSteps, region, minResolvable = 20):

    stepsCleaned = rawSteps.copy()
    spacings = spacingsFromSteps(rawSteps)
    #print(f"Filtered Double checking: {spacings}")
    
    
    noDoublesFound = True

    while noDoublesFound:  
        removeList = []
        noDoublesFound = False
        
        spacings = spacingsFromSteps(stepsCleaned)
        
        for i in range(0, len(spacings)-2):
            if spacings[i] > minResolvable:
                if spacings[i+1] <= spacings[i] and spacings[i+2]>= spacings[i+1]: #A double band on the assumption that doubles occur slightly left skewed of centered
                    #print(f"Potential Double, {spacings[i]} {spacings[i+1]} {spacings[i+2]}  {stepsCleaned[i+1]}, {region[stepsCleaned[i+1]]}, {region[stepsCleaned[i+2]]}, {region[stepsCleaned[i+3]]}")
                    if region[stepsCleaned[i+2]] < region[stepsCleaned[i+1]] and region[stepsCleaned[i+2]] < region[stepsCleaned[i+3]]: #Double must be weaker than both other sides
                        removeList.append(i+1)
       # print(removeList)
        
        
        for i in range(len(removeList), 0, -1):
            noDoublesFound = True
            stepsCleaned.pop(removeList[i-1]+1)

    if len(stepsCleaned) > 2:
        spacings = spacingsFromSteps(stepsCleaned)
        if spacings[-1] < 0.96* spacings[-2]:
            stepsCleaned.pop(-1)#Ends on a downturn edge case
        
    return stepsCleaned

    
def locateBands(region, sensitivity = 0.25, topEnd = 0.95, botEnd = 0.25, mode = "average", channel = 1, preProcessFront = True, preReject = 1.85, verbose = False, procDoubles = True, superscale = 1):
    if superscale != 1:
        region = skimage.transform.rescale(region, superscale, anti_aliasing=True)

    
    averageSlice = np.average(region,axis=1)

    data1D = []
    
    if mode == "average":
        data1D = np.average(averageSlice, axis =1 )
    elif mode == "channel":
        data1D = averageSlice[i][:,channel]

    dataNormed = data1D - np.min(data1D)
    dataNormed = dataNormed / np.max(dataNormed)

    gauss_filtered = gaussian_filter1d(dataNormed, 3, order=1)
    gauss_filtered = gauss_filtered / np.max(gauss_filtered)

    positiveRegion = np.clip(gauss_filtered, 0, 1.0)
    positiveRegion = positiveRegion/max(positiveRegion)
    negativeRegion = np.clip(gauss_filtered, -1.0, 0.0)
    negativeRegion = negativeRegion/min(negativeRegion)

    posSteps = step_detect.find_steps(positiveRegion, sensitivity)
    negSteps = step_detect.find_steps(negativeRegion,  0.10)


    N = len(data1D)
    posSteps = [x for x in posSteps if x < N * topEnd] 
    posSteps = [x for x in posSteps if x > N * botEnd] 
    negSteps = [x for x in negSteps if x < N * topEnd] 
    negSteps = [x for x in negSteps if x > N * botEnd] 
    
    if verbose:
        print(posSteps)
        #print(negSteps)

    if procDoubles:
        posSteps = filterDoubles(posSteps, positiveRegion)
    #negSteps = filterDoubles(negSteps) #We take specifically the nearest of the bottom edges so we dont need to reject doubles for both top and bottom, just one suffices (both caused issues)
    if verbose:
        print(f"After double filter: {posSteps}")
        #print(negSteps)
        
    
    if preProcessFront:
        posStepsProc = preprocessSteps(posSteps, rejection = preReject, rejectLast = False)
    else:
        posStepsProc = posSteps
        

    candidateBottoms = []
    failedIter = True

    while failedIter:
        candidateBottoms = []
        failedIter = False
        
        for posStepInx in range(0, len(posStepsProc)):
            if not failedIter:
                posStep = posStepsProc[posStepInx]
                candidates = []
                for negStepInx in range(0, len(negSteps)):
                    negStep = negSteps[negStepInx] 
        
                    limit = N-1
                    if posStepInx < (len(posStepsProc)-1): #not at end
                        limit = posStepsProc[posStepInx+1]
                    if negStep > posStep and negStep < limit:
                        candidates.append(negStep)
                if len(candidates) > 0:
                    candidateBottoms.append(candidates[-1])

                elif posStepInx == (len(posStepsProc)-1):
                    posStepsProc.pop(posStepInx)
                    failedIter = False #Last point irrelevant
                else: # No bottom was found between this and next top, remove that top and try again
                    posStepsProc.pop(posStepInx+1)
                    failedIter = True

    
    topEdges = posStepsProc[:-1] #always trim the last one
    botEdges = candidateBottoms[:-1]
    bandwidths = []
    for inx, top in enumerate(topEdges):
        bandwidths.append(botEdges[inx] - top)      

    if verbose:
        return positiveRegion, negativeRegion, topEdges, botEdges, bandwidths, posSteps, negSteps
    else:
        return positiveRegion, negativeRegion, topEdges, botEdges, bandwidths
