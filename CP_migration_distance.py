import numpy as np
import os, csv, glob
import matplotlib.path as mplPath
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.optimize import fmin_cobyla
import paths, exclFiles
import math


################################################################################
#  CUSTOM FUNCTIONS  
################################################################################


###     Extracting point coordinates of ROIs and neurons from txt files


#Function extracting neuron coordinates from txt files
def neuronsExtract(fileName):
    points = np.loadtxt(fileName, dtype='int', skiprows=1)
    return points[:,[1,2]][points[:,3]!=2]


#Functiong extracting cortical plate ROI
def cpExtract(fileName):
    points = np.loadtxt(fileName, dtype='int')
    return points  


#Function extracting only CP neurons
def neuronsCP(neurons, cp):
    
    #polygon used in the loop for extracting CP neurons
    polygon = mplPath.Path(cp)
    
    #list of CP neurons
    neuronsCP = []

    for x,y in neurons:
        if polygon.contains_point([x,y]): neuronsCP.append([x,y])
    return neuronsCP


###    Manipulating cortical plate coordinates


#Function extracting layers of speciment
def boundariesExtract(points):

    #indeces of points on edges of image
    iZeroX = points[:,0] == 0
    iZeroY = points[:,1] == 0
    iMaxX = points[:,0] == np.amax(points[:,0])
    iMaxY = points[:,1] == np.amax(points[:,1])

    #keep poins NOT on edges of image
    iKeep = np.logical_not(iZeroX + iZeroY + iMaxX + iMaxY)
    
    #return only points that mark boundaries of CP
    return points[iKeep]


#Function dividing set of points marking CP boundaries to apical and basal
def apicalBasalCluster(boundaries):
    
    #scale data for clustering algorithm
    X = StandardScaler().fit_transform(boundaries)
    
    #perform clustering on scaled data
    for n in range(1,10):
        db = DBSCAN(eps = (n * 0.1), min_samples=5).fit(X)
        labels = db.labels_
        unique_labels = set(labels)
        if (len(unique_labels) == 2): break

    if len(unique_labels) != 2: 
        print('Number of clusters DOES NOT equal 2!!!')
        
        #plot failure of designating 2 clusters
        plt.scatter(boundaries[:,0], boundaries[:,1], color = 'r')
        unique_labels = set(labels)


        colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
        
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = boundaries[class_member_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize = 8)
                
            plt.show()
          
        #interrupt the whole sequence              
        return False

    #divide points of boundaries to apical and basal and return    
    return [boundaries[labels == k] for k in unique_labels]
        

### Funcion rotating a point counterclockwise by a given angle in degrees around point (0,0):
### Modified form Alex Rodriguez post answer:
### https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point/18683594
def rotate(point, angle):

    px, py = point
    rads = math.radians(angle)

    qx = math.cos(rads) * (px) - math.sin(rads) * (py)
    qy = math.sin(rads) * (px) + math.cos(rads) * (py)
    
    #returns coordinates of rotated point
    return [qx, qy]


# Function converting boundaries to polynomial curves        
def apicalBasalCurveFit(bounds):
    
    #setting sum of sums of squares to be minimized to initial infinity
    rsq = float("inf")

    #rotating points -45 to +45 degrees every 15 degrees
    for deg in range(-45, 46, 15):
        
        #extracting 2 lists and rotating 
        rbounds1 = np.asarray([rotate(i, deg) for i in bounds[0]])
        rbounds2 = np.asarray([rotate(i, deg) for i in bounds[1]])
        
        #decide which list is apical and which basal
        if np.mean(rbounds1[:,1])>np.mean(rbounds2[:,1]):
            apical, basal = rbounds1, rbounds2
        else: 
            apical, basal = rbounds2, rbounds1
            
        #extracting point coordinates
        xa = apical[:,0]
        ya = apical[:,1]
        
        xb = basal[:,0]
        yb = basal[:,1]
        
        #fitting curves y = ax^2 + bx + c and saving corresponding sum of squares       
        za, resa, _, _, _ = np.polyfit(xa, ya, 2, full=True)
        zb, resb, _, _, _ = np.polyfit(xb, yb, 2, full=True)

        #if have better fit then previous ones, storing functions and new sum of sums of squares
        if rsq > resa + resb: 
            bestFit = [np.poly1d(za), np.poly1d(zb), deg] 
            rsq = resa + resb

    #returning functions with the best fit and corresponding rotation angle in degrees
    return bestFit



###    Calculating distance of neurons from apical and basal CP bounds
            
    
# Function finding closest points on apical and basal curves to specified neuron
# args: neuron x, y coordinates, list of 2 polynomial equations denoting
# cortical layer boundaries
# returns: neuron coordinates,
# its nearest points coordinates on apical and basal cortical plate boundaries
# distances form neurons to nearests points
# and relative neuron distance between apical and basal boundary of CP
def apicalBasalPts(neuron, apicalBasalCurves):
    aF = apicalBasalCurves[0]
    bF = apicalBasalCurves[1]
    deg = apicalBasalCurves[2]
    
    #rotating neuron as curves fitted to boundaries
    n = rotate(neuron, deg)
    
    #minimizing distance between points
    def objective(X):
        x,y = X
        return np.sqrt((x - n[0])**2 + (y - n[1])**2)
    
    #constrain definitions    
    def apical(X):
        x,y = X
        return y - aF(x)
        
    def basal(X):
        x,y = X
        return bF(x) - y
    
    #initial guess definition - start from the x coordinate of neuron
    def guess(f):
        return [n[0],f(n[0])]

    #minimization algorithm
    X = fmin_cobyla(objective, guess(aF), cons=[apical], disp = 0,
        maxfun = 10000, rhoend = 0.0001)
    aP = [X[0],X[1]]
    
    X = fmin_cobyla(objective, guess(bF), cons=[basal], disp = 0,
        maxfun = 10000, rhoend = 0.0001)
    bP = [X[0],X[1]]
    
    #rotating back coordinates of a neuron and its nearest points
    #on apical and basal surfaces of the CP
    n = rotate(n, -deg)
    aP = rotate(aP, - deg)
    bP = rotate(bP, -deg)
    
    #calculating distances of neuron to apical and basal surfaces
    #and its relative migration distance
    aD = np.sqrt((aP[0] - n[0])**2 + (aP[1] - n[1])**2)
    bD = np.sqrt((bP[0] - n[0])**2 + (bP[1] - n[1])**2)
    relDist = aD / (aD + bD)

    return [n, aP, bP, aD, bD, relDist]
    
    

################################################################################
###    MAIN PART OF SCRIPT
################################################################################

os.chdir(paths.inputDir)
fileList = glob.glob("*neurons.txt")
specList = sorted(set([fileList[i][:-12] for i in range(len(fileList))]))

quantList = [s for s in specList if s not in exclFiles.skipList]

#empty list for storing summary results of mean migration distance in cortical plate
cpRes = []

#empty list for storing results of single neuron migration distance in cortical plate
singNeurMigr = []

#Main loop iterating through speciments in a inputDir writing to a file in outputDir
for specName in quantList:
    print('adding results for ' + specName)

    #isolate cortical plate region
    cp = cpExtract(paths.inputDir + specName + '_ROI-2.txt')
    cpBounds = boundariesExtract(cp)
    #cluster points of apical and basal CP boundaries
    ab = apicalBasalCluster(cpBounds)
    #fit curves into apical and basal CB boundaries and get rotation degree
    cAB = apicalBasalCurveFit(ab)
    
    allNeurons = neuronsExtract(paths.inputDir + specName + '_neurons.txt')
    #extract only neurons within CP
    cpNeurons = neuronsCP(allNeurons, cp)
    
    #save results for all neurons of a speciment to a list
    specRes = [apicalBasalPts(n, cAB) for n in cpNeurons]
    
    #append apical/basal migration ratio for all neurons of speciment to a global list
    for sr in specRes:
        singNeurMigr.append([specName, sr[5]])

    #write all point coordinates, distances and ratio for a speciment to a separate file
    with open(paths.outputDirCP + specName + '_CPmigr.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(specRes)
        
        
    ### VISUALISATION OF EACH SPECIMENT
    
    #using section image as background for plotting
    img = plt.imread(paths.inputDirImg + "MaxProject_" + specName + '.tif')
    fig, ax = plt.subplots()
    axes = plt.gca()
    axes.set_xlim([0,img.shape[1]])
    axes.set_ylim([img.shape[0],0])
    plt.axis('off')
    ax.imshow(img)

    x_new = np.linspace(0, np.amax(cp[:,0]), np.amax(cp[:,0]))
    
    #show neurons in yellow, distance to apical in purple, to basal in magenta
    xNs, yNs = np.array(allNeurons)[:,0], np.array(allNeurons)[:,1]
    plt.scatter(xNs, yNs, color = 'pink', s = 1)
    
    xCPNs, yCPNs = np.array(cpNeurons)[:,0], np.array(cpNeurons)[:,1]
    plt.scatter(xCPNs, yCPNs, color = 'yellow', s = 1)
    
    #show distances from neuron to apical in blue, to basal in red
    APs, BPs = np.array(specRes)[:,1], np.array(specRes)[:,2]

    for i in range(len(xCPNs)):
        plt.plot([APs[i][0],xCPNs[i]], [APs[i][1],yCPNs[i]], color = 'b',  linewidth=0.5)
        plt.plot([BPs[i][0],xCPNs[i]], [BPs[i][1],yCPNs[i]], color = 'r',  linewidth=0.5)
    plt.text(0, 100, specName, fontsize=12, color = 'white')
    fig.savefig(paths.outputDirCPImg + specName + '.jpg', dpi = 300,  bbox_inches='tight')
    

    ### STORING SUMMARY RESULTS

    #converting list of lists with results to numpy array and extracting last colum
    #containing relative distance of CP migration and converting to float    
    aRelDist = np.array(specRes)[:,5].astype(float)
    #calculate mean relative distance of neural progenitor migration in cortical plate
    #for a speciment and append to a list storing summary results
    cpRes.append([specName, np.mean(aRelDist)])

#write to a file summary results of mean relative distance of CP migration
with open(paths.outputDirCP + 'outputCPmigr.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(cpRes) 


with open(paths.outputDirCP + 'outputCPmigrSINGLE.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(singNeurMigr)    