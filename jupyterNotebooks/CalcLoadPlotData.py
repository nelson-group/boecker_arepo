import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from os.path import exists

yr_to_sec = 60 * 60 * 24 * 365
sec_to_yr = 1 / yr_to_sec
cm_to_km = 1 / 1e5
cm_to_pc = 1 / 3.085678e18

#for temperature calculation:
kB_SI = 1.38e-23 #J/K
kB_cgs = kB_SI * 1e7
X_H = 0.76 # hydrogen mass fraction
gamma = 5/3
m_p = 1.67e-24 # proton mass in g


UnitVelocity_in_cm_per_s = 1e5		
UnitVelocity_in_km_per_s = UnitVelocity_in_cm_per_s / 1e5
UnitLength_in_cm = 3.085678e18 	# pc
UnitLength_in_pc = UnitLength_in_cm / 3.085678e18
UnitLength_in_kpc = UnitLength_in_pc / 1000
UnitMass_in_g = 1.989e31		# 0.01 solar masses
UnitMass_in_Msun = UnitMass_in_g / 1.989e33

UnitEnergy_in_erg = UnitMass_in_g * UnitVelocity_in_cm_per_s**2
UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s
UnitTime_in_yr = UnitTime_in_s * sec_to_yr
UnitDensity_in_cgs = UnitMass_in_g / UnitLength_in_cm**3


boxSize = 96
frameAmount = 40
frameNbrMultiplier = 10
TimeBetSnapshot_in_unit_time = 0.002

def GetParameters():
    print("boxSize = "+ str(boxSize) +" \nframeAmount = "+ str(frameAmount) +" \nframeNbrMultiplier = "+ str(frameNbrMultiplier) +" \nTimeBetSnapshot_in_unit_time = "+ str(TimeBetSnapshot_in_unit_time))
    
def SetParameters(_boxSize = -1, _frameAmount = -1, _frameNbrMultiplier = -1, _TimeBetSnapshot_in_unit_time = -1):
    if(_boxSize != -1):
        global boxSize 
        boxSize  = _boxSize
    if(_frameAmount != -1):
        global frameAmount 
        frameAmount  = _frameAmount
    if(_frameNbrMultiplier != -1):
        global frameNbrMultiplier 
        frameNbrMultiplier  = _frameNbrMultiplier
    if(_TimeBetSnapshot_in_unit_time != -1):
        global TimeBetSnapshot_in_unit_time 
        TimeBetSnapshot_in_unit_time  = _TimeBetSnapshot_in_unit_time


def GetUnitSystem():
    print("UnitVelocity_in_cm_per_s = "+ str(UnitVelocity_in_cm_per_s) +" \nUnitLength_in_cm = "+ str(UnitLength_in_cm) +" \nUnitMass_in_g = "+ str(UnitMass_in_g))
    

def SetUnitSystem(_UnitVelocity_in_cm_per_s = -1, _UnitLength_in_cm = -1, _UnitMass_in_g = -1):
    if(_UnitVelocity_in_cm_per_s != -1):
        global UnitVelocity_in_cm_per_s 
        UnitVelocity_in_cm_per_s  = _UnitVelocity_in_cm_per_s
        global UnitVelocity_in_km_per_s
        UnitVelocity_in_km_per_s = UnitVelocity_in_cm_per_s / 1e5

    if(_UnitLength_in_cm != -1):
        global UnitLength_in_cm 
        UnitLength_in_cm = _UnitLength_in_cm
        global UnitLength_in_pc
        UnitLength_in_pc = UnitLength_in_cm / 3.085678e18
        global UnitLength_in_kpc
        UnitLength_in_kpc = UnitLength_in_pc / 1000

    if(_UnitMass_in_g != -1):
        global UnitMass_in_g
        UnitMass_in_g = _UnitMass_in_g
        global UnitMass_in_Msun
        UnitMass_in_Msun = UnitMass_in_g / 1.989e33

    global UnitEnergy_in_erg
    UnitEnergy_in_erg = UnitMass_in_g * UnitVelocity_in_cm_per_s**2
    global UnitTime_in_s
    UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s
    global UnitTime_in_yr
    UnitTime_in_yr = UnitTime_in_s * sec_to_yr
    global UnitDensity_in_cgs
    UnitDensity_in_cgs = UnitMass_in_g / UnitLength_in_cm**3
    


def LoadAnalyticalSolutions(nSN = 1):
    if(nSN == 1):
        with h5py.File("/vera/u/xboecker/arepo/jupyterNotebooks/analyticalSolution/maxRadVelData_80k_years.hdf5", "r") as hdf:
            radVelData = np.array(hdf.get("Data/RadVels"))
            timeData = np.array(hdf.get("Data/Times")) * 10
        return timeData, radVelData
        #with h5py.File("analyticalSolution/maxRadVelData_80k_years_1e51erg_p_1e-1.hdf5", "r") as hdf:
        #    radVelData = np.array(hdf.get("Data/RadVelsp1e-1"))
        #    timeData = np.array(hdf.get("Data/Timesp1e-1"))
        #return timeData, radVelData
    elif(nSN == 10):
        with h5py.File("analyticalSolution/maxRadVelData_80k_years_10_SNe.hdf5", "r") as hdf:    
            radVelData10 = np.array(hdf.get("Data/RadVels10"))
            timeData10 = np.array(hdf.get("Data/Times10"))
        return timeData10, radVelData10
    elif(nSN == 100):
        with h5py.File("analyticalSolution/maxRadVelData_80k_years_100_SNe.hdf5", "r") as hdf:    
            radVelData100 = np.array(hdf.get("Data/RadVels100"))
            timeData100 = np.array(hdf.get("Data/Times100"))
        return timeData100, radVelData100
    elif(nSN == 0.43):
        with h5py.File("analyticalSolution/maxRadVelData_80k_years_1e43erg.hdf5", "r") as hdf:    
            radVelData1e43 = np.array(hdf.get("Data/RadVels1e43"))
            timeData1e43 = np.array(hdf.get("Data/Times1e43"))
        return timeData1e43, radVelData1e43
    
    
    
    
        
# Load Data

def LoadDataFromHDF(folder, timeStep, dataName, partType = "PartType0"):
    with h5py.File(folder+"/snap_"+str(timeStep).zfill(3) +".hdf5", "r") as hdf:
        data = np.array(hdf.get(partType+"/"+dataName))
    return data

def getVelocities(folder, timeStep):
    with h5py.File(folder+"/snap_"+str(timeStep).zfill(3) +".hdf5", "r") as hdf:
        velocities = np.array(hdf.get("PartType0/Velocities"))
    return velocities

def getZVelocities(folder, timeStep):
    return getVelocities(folder, timeStep)[:,2]

def getMasses(folder, timeStep):
    with h5py.File(folder+"/snap_"+str(timeStep).zfill(3) +".hdf5", "r") as hdf:
        masses = np.array(hdf.get("PartType0/Masses"))
    return masses

def getStarMasses(folder, timeStep):
    with h5py.File(folder+"/snap_"+str(timeStep).zfill(3) +".hdf5", "r") as hdf:
        if "PartType4" not in hdf:
            return np.zeros(1)
        masses = np.array(hdf.get("PartType4/Masses"))
    return masses

def getDensities(folder, timeStep):
    with h5py.File(folder+"/snap_"+str(timeStep).zfill(3) +".hdf5", "r") as hdf:
        densities = np.array(hdf.get("PartType0/Density"))
    return densities

def getCoos(folder, timeStep):
    with h5py.File(folder+"/snap_"+str(timeStep).zfill(3) +".hdf5", "r") as hdf:
        coos = np.array(hdf.get("PartType0/Coordinates"))
    return coos

def getSFRs(folder, timeStep):
    with h5py.File(folder+"/snap_"+str(timeStep).zfill(3) +".hdf5", "r") as hdf:
        sfrs = np.array(hdf.get("PartType0/StarFormationRate"))
    return sfrs

def getStarsExploded(folder, timeStep):
    with h5py.File(folder+"/snap_"+str(timeStep).zfill(3) +".hdf5", "r") as hdf:
        try:
            hdf['PartType4']
        except KeyError:
            starsExploded = np.zeros(1)
        else:
            starsExploded = np.array(hdf.get("PartType4/snCount"))
            #starsExploded = np.array(hdf.get("PartType4/StarsExploded"))
            if(starsExploded.all() == None):
                starsExploded = np.zeros(1)
    return starsExploded

def T(internalEnergy, electronAbundance): #input in codeunits output in Kelvin
    x_e = electronAbundance
    u = internalEnergy
    
    my = 4/(1 + 3 * X_H + 4 * X_H * x_e) * m_p # meanMolecularWeight
    T = (gamma - 1) * u / kB_cgs * UnitEnergy_in_erg / UnitMass_in_g * my
    return T

def getTemperaturesInKelvin(folder, timeStep, debugOn = False):
    energies = LoadDataFromHDF(folder, timeStep, "InternalEnergy")
    electronAbundance = LoadDataFromHDF(folder, timeStep, "ElectronAbundance")
    
    temperatures = T(energies, electronAbundance)
    
    if(timeStep == 0):
        print("initial temperature: " + str(temperatures[0]) + "K")
    if(debugOn):
        print("max temp: " + str(np.max(temperatures)))
        print("min temp: " + str(np.min(temperatures)))
    
    return temperatures


# Usefull stuff

def getRadialDistances(folder, timeStep):
    coos = LoadDataFromHDF(folder, timeStep, "Coordinates")
    radDirs = coos - np.full(3,boxSize/2)
    radDirsNorm = np.sqrt((radDirs*radDirs).sum(axis=1))
    return radDirsNorm



# Velocity

def getRadialGasVelocities(folder, timeStep):
    vels = LoadDataFromHDF(folder, timeStep, "Velocities")
    coos = LoadDataFromHDF(folder, timeStep, "Coordinates")
    
    radDirs = coos - np.full(3,boxSize/2)
    radDirsNorm = np.sqrt((radDirs*radDirs).sum(axis=1))
    normedRadDirs = radDirs / radDirsNorm[:,None] # radDirs: (n,3); radDirsNorm: (n,1)

    radVels = vels * normedRadDirs
        
    return radVels

def getMaxRadialGasVelocity(folder, timeStep): # maxRadVels
    radVels = getRadialGasVelocities(folder, timeStep)
    return np.max(radVels)

def getAbsoluteVelocities(folder, timeStep):
    velocities = LoadDataFromHDF(folder, timeStep, "Velocities")
    absoluteVelocities = np.sqrt((velocities*velocities).sum(axis=1))
    return absoluteVelocities


def getMaxGasVelocity(folder, timeStep):    
    absVelocities = getAbsoluteVelocities(folder, timeStep)
    return np.max(absVelocities)



# Momentum

def getMomenta(folder, timeStep):
    absVelocities = getAbsoluteVelocities(folder, timeStep)
    masses = getMasses(folder, timeStep)
    momenta = masses * absVelocities
    return momenta

def getTotalMomentum(folder, timeStep): # totalMomentum
    momenta = getMomenta(folder, timeStep)
    totalMomentum = np.sum(momenta)
    return totalMomentum




# Energy

def getMaxEnergy(folder, timeStep):
    internalEnergies = LoadDataFromHDF(folder, timeStep, "InternalEnergy")
    return np.max(internalEnergies)

def getTotalInternalEnergy(folder, timeStep):
    internalEnergies = LoadDataFromHDF(folder, timeStep, "InternalEnergy")
    return np.sum(internalEnergies)

def getTotalKineticEnergy(folder, timeStep):
    masses = LoadDataFromHDF(folder, timeStep, "Masses")
    absVelocities = getAbsoluteVelocities(folder, timeStep)
    kineticEnergies = 0.5 * masses * absVelocities*2
    return np.sum(kineticEnergies)

def getTotalEnergy(folder, timeStep):
    internalEnergy = getTotalInternalEnergy(folder, timeStep)
    kineticEnergy = getTotalKineticEnergy(folder, timeStep)
    totalEnergy = internalEnergy + kineticEnergy
    return totalEnergy

def getTotalStarsExploded(folder, timeStep):
    starsExploded = getStarsExploded(folder, timeStep)
    totalStarsExploded = np.sum(starsExploded)
    return totalStarsExploded

def getStarsExplodedThisTimeStep(folder, timeStep):
    totalStarsExploded = getTotalStarsExploded(folder, timeStep)
    if(timeStep == 0):
        return totalStarsExploded
    lastStepTotalStarsExploded = getTotalStarsExploded(folder, timeStep-1)
    return totalStarsExploded - lastStepTotalStarsExploded



# SNR Radius


# for ana solution, or loaded radVels

def getRadiiFromRadialVelocities(radVels, timeBetweenEntries): # input: array of max radVels of all timeSteps (for ana solution)
    radii = radVels
    for i in range(1, len(radVels)):
        radii[i] += radii[i-1]
    radii *= timeBetweenEntries
    return radii


# radius via integration of radVels

def getIncreasOfRadiusInThisTimeStep(radVels, timeBetweenEntries):
    return np.max(radVels) * timeBetweenEntries

def maxRadVelIntegrationRadius(folder, timeStep, oldRadius):
    radVels = getRadialGasVelocities(folder, timeStep)
    snrRadius = oldRadius + getIncreasOfRadiusInThisTimeStep(radVels, TimeBetSnapshot_in_unit_time)
    return snrRadius

# radius via particle-density profile maximum

def densityProfileRadius(folder, timeStep, bins):
    radDists = getRadialDistances(folder, timeStep)
    blastwave = np.zeros(len(bins))
    for i in range(len(coos)):
        for j in range(len(bins)-1):
            if(radDists[i] >= bins[j] and radDists[i] < bins[j+1]):
                blastwave[j] += 1#densities[i] / radDists[i] ** 2
                
    return np.where(blastwave == np.max(blastwave))[0][0]

# radius via point of max density

def maxDensityRadius(folder, timeStep): # maxDensityRadii
    coos = getCoos(folder, timeStep)
    densities = getDensities(folder, timeStep)
    highestDensityIndex = np.where(densities == np.max(densities))[0][0]
    radius = np.sqrt((coos[highestDensityIndex,0] - boxSize/2)**2 + 
                     (coos[highestDensityIndex,1] - boxSize/2)**2 + 
                     (coos[highestDensityIndex,2] - boxSize/2)**2)
    return radius

# radius via velocity cut

def velocityCutRadius(folder, timeStep, debugOn = False): # velocityCutRadii
    absVelocities = getAbsoluteVelocities(folder, timeStep)
    distsFromCenter = getRadialDistances(folder, timeStep)
    
    if(debugOn):
        print(f"{absVelocities}")
        print(f"{distsFromCenter}")
    
    velocityCut_in_km_per_s = 1
    velocityCut_in_Unit_velocity = velocityCut_in_km_per_s / UnitVelocity_in_km_per_s
    particlesWithEnoughVelocity = distsFromCenter[np.where(absVelocities > velocityCut_in_Unit_velocity)]
    
    if(len(particlesWithEnoughVelocity) > 0):
        radius = np.max(particlesWithEnoughVelocity)
        return radius
    else:
        return 0
    






# input radii of all timesteps

def shellVelocitiesFromRadii(radii):
    shellVelocities = np.zeros(len(radii))
    shellVelocities[0] = radii[0] * UnitLength_in_cm / 1e3 / TimeBetSnapshot_in_s
    shellVelocities[1:] = (radii[1:] - radii[:-1]) * UnitLength_in_cm / 1e3 / TimeBetSnapshot_in_s
    
    return shellVelocities # in km/s

def RadiiFromShellVelocities(vels):
    radii = vels * TimeBetSnapshot_in_unit_time
    for i in range(len(vels)):
        if(i == 0):
            continue
        radii[i] += radii[i-1]
    return radii



def getTotalSFR(folder, timeStep): # SFR
    sfrs = getSFRs(folder, timeStep)
    #print(sfrs)
    totalSFR = np.sum(sfrs)
    #print(totalSFR)
    return totalSFR










def getTotalOutflowMass(folder, timeStep):
    velocities = getVelocities(folder, timeStep)
    coos = getCoos(folder, timeStep)
    masses = getMasses(folder, timeStep)
    
    xy_velocity_mag = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
    
    upperOutflowIndices = np.where((velocities[:, 2] > xy_velocity_mag) & (coos[:, 2] > 100))
    upperOutflowMass_sum = np.sum(masses[upperOutflowIndices])
    
    lowerOutflowIndices = np.where((velocities[:, 2] < -xy_velocity_mag) & (coos[:, 2] < 100))
    lowerOutflowMass_sum = np.sum(masses[lowerOutflowIndices])
    
    totalOutflowMass_sum = upperOutflowMass_sum + lowerOutflowMass_sum
    
    return totalOutflowMass_sum

def getTotalInflowMass(folder, timeStep):
    velocities = getVelocities(folder, timeStep)
    coos = getCoos(folder, timeStep)
    masses = getMasses(folder, timeStep)
    
    xy_velocity_mag = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
    
    upperInflowIndices = np.where((velocities[:, 2] < -xy_velocity_mag) & (coos[:, 2] > 100))
    upperInflowMass_sum = np.sum(masses[upperInflowIndices])
    
    lowerInflowIndices = np.where((velocities[:, 2] > xy_velocity_mag) & (coos[:, 2] < 100))
    lowerInflowMass_sum = np.sum(masses[lowerInflowIndices])
    
    totalInflowMass_sum = upperInflowMass_sum + lowerInflowMass_sum
    
    return totalInflowMass_sum


def getTotalOutflowMinusInflow(folder, timeStep):
    outflow = getTotalOutflowMass(folder, timeStep)
    inflow = getTotalInflowMass(folder, timeStep)
    outflowMinusInflow = outflow - inflow
    
    return outflowMinusInflow

def getTotalDiskMass(folder, timeStep):
    masses = getMasses(folder, timeStep)
    coos = getCoos(folder, timeStep)
    starMasses = getStarMasses(folder, timeStep)
    
    diskIndices = np.where((coos[:,2] < 114) & (coos[:,2] > 96) & (np.sqrt((coos[:,0]-100)**2 + (coos[:,1]-100)**2) < 25))
    
    diskMasses = masses[diskIndices]
    
    return np.sum(diskMasses) + np.sum(starMasses)





def calculateOrLoadData(folder, folderName, dataName, calcDataOfOneSnapShotFunction, frameAmount, recalculate = False):
    if(exists("/vera/u/xboecker/jupyterNotebooksOutputs/data/"+dataName+folderName) and recalculate == False):
        #load data
        print("load data")
        data = np.loadtxt("/vera/u/xboecker/jupyterNotebooksOutputs/data/"+dataName+folderName)
        if(len(data) < frameAmount):
            print("create more data")
            newData = np.zeros(frameAmount)
            
            dh = display(0,display_id=True)
            
            for timeStep in range(frameAmount):
                if(timeStep < len(data)):
                    newData[timeStep] = data[timeStep]
                else:
                    newData[timeStep] = calcDataOfOneSnapShotFunction(folder, timeStep)
                dh.update(timeStep)
                np.savetxt("/vera/u/xboecker/jupyterNotebooksOutputs/data/"+dataName+folderName, newData)
            return newData
        else:
            return data[:frameAmount]
    else:
        if(recalculate):
            print("recalculate data")
        else:
            print("calculate data")
        data = np.zeros(frameAmount)
        
        dh = display(0,display_id=True)
            
        #calculate and save new data
        for timeStep in range(frameAmount):
            data[timeStep] = calcDataOfOneSnapShotFunction(folder, timeStep)
            dh.update(timeStep)

            np.savetxt("/vera/u/xboecker/jupyterNotebooksOutputs/data/"+dataName+folderName, data)
        return data
    

def PlotData(data, folderNames, title, ylabel, dataName, unit_conversion_factor, frameAmount, frameNbrMultiplier, TimeBetSnapshot_in_unit_time, xScaleLog = False, xlabel = "time [yr]", legendLabels = [""], compareToAnalyticVelocities = False, compareToAnalyticRadius = False):
    plt.clf()
    
    TimeBetSnapshot_in_s = TimeBetSnapshot_in_unit_time * UnitTime_in_s
    TimeBetSnapshot_in_yr = TimeBetSnapshot_in_s * sec_to_yr
    timeScaleToYears = np.linspace(0,TimeBetSnapshot_in_yr*(frameAmount-1),frameAmount)

    if(xScaleLog):
        #print(timeScaleToYears[1:frameAmount])
        timeScaleToYears = np.log10(timeScaleToYears[1:frameAmount] / 1e6)
        timeScaleToYears = np.append(timeScaleToYears[0], timeScaleToYears)
        xlabel = "time [log10(Myr)]"
    
    print(timeScaleToYears)
    for i in range(len(folderNames)):
        plt.plot(timeScaleToYears[1:frameAmount]*frameNbrMultiplier, data[i][1:frameAmount] * unit_conversion_factor, label=folderNames[i])
    
    if(compareToAnalyticVelocities):
        timeData, radVelData = LoadAnalyticalSolutions()
        plt.plot(timeData * sec_to_yr, radVelData * cm_to_km, label="Sedov Taylor Radial Velocity")
    elif(compareToAnalyticRadius):
        timeData, radVelData = LoadAnalyticalSolutions()
        radii = RadiiFromShellVelocities(radVelData)
        plt.plot(timeData * sec_to_yr, radii * cm_to_km, label="Sedov Taylor Shell Radius")

    plt.yscale("log")
    plt.title(title)
    
    if(legendLabels == [""]):
        plt.legend()
    else:
        plt.legend(legendLabels)
    plt.style.use('./my_style.mplstyle')
    #plt.legend(["1 star center", "10 stars center", "100 stars center", "10 stars flat dist", "100 stars flat dist", "analytical 1 SN", "analytical 10 SNe", "analytical 100 SNe"])
    #plt.legend(["1 star", "10 stars r=10pc", "10 stars t=5kyr", "10 stars r=10pc t=5kyr"])
    plt.xlabel(xlabel)
    #plt.ylabel("Max velocity ["+str(round(UnitVelocity_in_km_per_s))+ " km/s]")
    plt.ylabel(ylabel)
    #fig = plt.figure()#figsize=(16.18 * 2, 10 * 2))
    #fig.set_dpi(150.0)
    #fig.patch.set_facecolor('xkcd:mint green')
    saveName = "/vera/u/xboecker/jupyterNotebooksOutputs/plots/"+dataName
    for i in range(len(folderNames)):
        saveName += "-"+folderNames[i]
        
    saveName += ".png"
    plt.savefig(saveName)#, facecolor='w')

    plt.show()
    
    
    
def PlotDensityTempPhaseDiagramm(folder, timeStep, galaxyBoxSize, savePath):
    densities = getDensities(folder, timeStep)*UnitDensity_in_cgs /m_p
    temperatures = getTemperaturesInKelvin(folder, timeStep)
    masses = getMasses(folder, timeStep)
    radDistances = getRadialDistances(folder, timeStep, galaxyBoxSize) * UnitLength_in_kpc
    diskDensities = densities[np.where(radDistances < 20)]
    diskTemperatures = temperatures[np.where(radDistances < 20)]

    #print(np.min(densities))
    #print(np.max(densities))
    #print(np.min(temperatures))
    #print(np.max(temperatures))
    fig, ax = plt.subplots(1,1, figsize=(13,8))
    
    ax.hist2d(np.log10(densities), np.log10(temperatures), weights=masses, bins=50, norm = mpl.colors.LogNorm(), range=[[-8,1],[0,9]])
    ax.set_xlabel("density [cm-3]")
    ax.set_ylabel("temperature [K]")
    filename = savePath + 'frame_%03d.png' % timeStep
    fig.savefig(filename)
    plt.close(fig)
