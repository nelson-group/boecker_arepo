/*!
 * \copyright   This file is part of the public version of the AREPO code.
 * \copyright   Copyright (C) 2009-2019, Max-Planck Institute for Astrophysics
 * \copyright   Developed by Volker Springel (vspringel@MPA-Garching.MPG.DE) and
 *              contributing authors.
 * \copyright   Arepo is free software: you can redistribute it and/or modify
 *              it under the terms of the GNU General Public License as published by
 *              the Free Software Foundation, either version 3 of the License, or
 *              (at your option) any later version.
 *
 *              Arepo is distributed in the hope that it will be useful,
 *              but WITHOUT ANY WARRANTY; without even the implied warranty of
 *              MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *              GNU General Public License for more details.
 *
 *              A copy of the GNU General Public License is available under
 *              LICENSE as part of this program.  See also
 *              <https://www.gnu.org/licenses/>.
 *
 * \file        src/xeno_sn.c
 * \date        03/2022
 * \brief       Handling of SNe.
 * \details     Checks if star is ready to explode, then does thermal dump.
 *              contains functions:
 *                void SetupStarBirthTimes()
 *                void CalcStarAges()
 *
 */

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//#include "gravity/forcetree.h"
#include "main/allvars.h"
#include "main/proto.h"

#ifdef XENO_SN

float starLifeTime = 10 * 1e6 * 365 * 24 * 60 * 60; // 10 million years in seconds

double sunMass_in_g    = 1.989e33;
double snePerMSun = 0.01;
double massEjectedPerSN_in_g  = 1.989e33;
double snEnergyToInputInErgs     = 1e51;
int amountofcellstodepositenergy = 8;

int N_explosions = 0;
int allTasks_N_explosions = 0;
int thisTaskSNStarParticleIndices[100000];
int allTasks_SNStarParticleIndices[100000];
float thisTaskSNPositions[1000000][3]; // space for maxSNCountPerTimestep SNe on every task
float allTasks_SN_positions[1000000][3];
int maxSNCountPerTimestep = 1000000;
int allTasksCounts[1000000];

int closestCellIndices[1000000]; // max length needed: amountofcellstodepositenergy
float closestDistances[1000000]; // max length needed: amountofcellstodepositenergy

float allTasks_ClosestDistances[1000000]; // max length needed: amountofcellstodepositenergy * NTask

int test = 0;

#pragma region functionDeclarations
int IsExistingStarParticle(int i);
int IsLivingStar(int i);
int SNHappened();
void ExplodeStar(int i);
void GetSNCountAndPositionsOnThisTask();

void CommunicateSNCountsAmongAllTasks();

void SetAllGasCellsActive();

void CommunicateSNPositionsAndStarParticleIndicesAmongAllTasks();

void FindClosestCellsOnThisTask(int currentSNIndex);

void CommunicateClosestCellsOfAllTasks();

void FindAllTaskClosestCellsOnThisTask();

void AddEnergyAndMassToAllTasksClosestCells();

void SetAllParticlesTimeStepToSmallestTimestep();

#pragma endregion

// don't forget to add fcts to proto.h (if you want to call them from somewhere else)

#pragma region Tests

void TestPositionsAndDistances(){
  for(int i = 0; i < NumGas; i++){
    float x = P[i].Pos[0];
    float y = P[i].Pos[1];
    float z = P[i].Pos[2];

    float squareDistToCell = 0;
    for(int j = 0; j < 3; j++)
    {
      float dist = P[i].Pos[j] - All.BoxSize/2;
      
      //if(dist > (All.BoxSize/2)) dist = All.BoxSize - dist;

      squareDistToCell += dist * dist;
    }

      printf("Task (%d): cellID: %d Distance: %g, position: x = %g, y = %g, z = %g\n", ThisTask, P[i].ID, squareDistToCell, x , y , z);

  }

  terminate("Test end");
}

void TestGetSNInjectCellIDs(){

  for(int i = 0; i < NumGas; i++){
    if(SphP[i].Utherm > SphP[0].Utherm * 10){
      float x = P[i].Pos[0];
      float y = P[i].Pos[1];
      float z = P[i].Pos[2];

      float squareDistToCell = 0;
      for(int j = 0; j < 3; j++)
      {
        float dist = P[i].Pos[j] - All.BoxSize/2;
        
        if(dist > (All.BoxSize/2)) dist = All.BoxSize - dist;

        squareDistToCell += dist * dist;
      }

      double cellsUtherm = SphP[i].Utherm * (SphP[i].OldMass * All.UnitMass_in_g) * All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s;
      printf("...Cell with %g erg Energy (%lf EnergyInCodeUnits), mass: %g, ID: %d, Distance: %g, position: x = %g, y = %g, z = %g\n", cellsUtherm, SphP[i].Utherm, SphP[i].OldMass, P[i].ID, squareDistToCell, x, y, z);
    }
  }
}


void TestRepeatICSN(){ // adds SN energy to the test Cells

    int count = 0;
  for(int i = 0; i < NumGas; i++){

    if(P[i].ID == 1007292){
      printf("Particle 1007292 startPosition:\n");
      for(int j = 0; j < NumGas; j++){
        printf("%g", P[i].Pos[j]);
      }
    }
    if(P[i].ID == 1007420){
      printf("Particle 1007420 startPosition:\n");
      for(int j = 0; j < NumGas; j++){
        printf("%g", P[i].Pos[j]);
      }
    }


    float squareDistToCell = 0;
    for(int j = 0; j < 3; j++)
    {
      float dist = P[i].Pos[j] - All.BoxSize/2;
      
      //if(dist > (All.BoxSize/2)) dist = All.BoxSize - dist;

      squareDistToCell += dist * dist;
    }
    /*
    if(squareDistToCell < 0.2){
      count++;
      printf("centerCellFound at squaredistance: %f: depositing Energy!\n", squareDistToCell);
      double SpecificEnergyToAdd      = snEnergyToInputInErgs / (SphP[i].OldMass * All.UnitMass_in_g);
      double conversionErgToCodeunits = All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s;
      double energyToAddInCodeUnits   = SpecificEnergyToAdd / conversionErgToCodeunits;


      SphP[i].Utherm += energyToAddInCodeUnits / amountofcellstodepositenergy;
      SphP[i].Energy += All.cf_atime * All.cf_atime * energyToAddInCodeUnits * P[i].Mass /
                                      amountofcellstodepositenergy;
    }*/
    
  }
  printf("%d cells found", count);
}


void Test(){
  //TestPositionsAndDistances();
  //TestGetSNInjectCellIDs();
  //TestRepeatICSN();
}

#pragma endregion

void SetupNewStar(int i)
{
  //printf("Setup new Star");
  P[i].StarBirthTime = All.Ti_Current * All.Timebase_interval * All.UnitTime_in_s;
  P[i].StarLifeTime = starLifeTime * get_random_number_aux();
  P[i].StarExploded = 0;
}

void ExplodeAllDyingStars()
{
    
    //if(test > 0){
    //  
    //  Test();
    //  test--;
    //}

    // if(All.NumCurrentTiStep >= 10) terminate("lets stop.");

    GetSNCountAndPositionsOnThisTask();

    CommunicateSNCountsAmongAllTasks();
    
    if(!SNHappened()) return;

    if(ThisTask == 0) printf("\n\nSN Happened!!!!! \n\n");

    // SetAllGasCellsActive();

    CommunicateSNPositionsAndStarParticleIndicesAmongAllTasks();

    // find closest cells of all task and deposit energy for each SN that happened this timestep
    for(int currentSNIndex = 0; currentSNIndex < allTasks_N_explosions; currentSNIndex++)
    {
        FindClosestCellsOnThisTask(currentSNIndex);

        CommunicateClosestCellsOfAllTasks();
        
        FindAllTaskClosestCellsOnThisTask();

        AddEnergyAndMassToAllTasksClosestCells(currentSNIndex);
    }

    #ifdef SN_TIMESTEPS
    adjust_timestep_of_all_particles_to_next_sn();
    #endif
} // SNe done


#pragma region helperFunctionsForCalcStarAges

int IsLivingStar(int i){
    if (P[i].Mass == 0 && P[i].ID == 0)
        return 0;         /* skip cells that have been swallowed or eliminated */
    if (P[i].Type != 4)  // skip non star cells
        return 0;
    if(P[i].StarExploded == 1)
        return 0;
    //TODO: check if star is actually active --> skip ( return 0)

    return 1;
}

int SNHappened(){
    allTasks_N_explosions = 0;
    for(int i = 0; i < NTask; i++){
        if(allTasksCounts[i] > 0) allTasks_N_explosions += allTasksCounts[i];
    }

    if(allTasks_N_explosions > 0) printf("\nallTasks_N_explosions = %d", allTasks_N_explosions);

    if(allTasks_N_explosions == 0) return 0;

    return 1;
}

void ExplodeStar(int i){
  if(N_explosions >= maxSNCountPerTimestep) terminate("More than %d SNe on one task at the same time (increase thisTaskSNPositions array size)", maxSNCountPerTimestep);

    // add new SN position into new array
    for(int j = 0; j < 3; j++){
        thisTaskSNPositions[N_explosions][j] = P[i].Pos[j];
        thisTaskSNStarParticleIndices[N_explosions] = i;
    }

    N_explosions++;

    P[i].StarExploded         = 1;

    //printf("\n\n\n\n\nexplode!\n\n\n");
}

void GetSNCountAndPositionsOnThisTask(){
  N_explosions = 0;
  for(int i = 0; i < maxSNCountPerTimestep; i++){
    for(int j = 0; j < 3; j++){
      thisTaskSNPositions[i][j] = 0;
    }
  }
  for(int i = NumGas; i < NumPart; i++)  // skip gas particles
    {
        if(!IsLivingStar(i)) continue;
        double timeInUnitTimes = All.Ti_Current * All.Timebase_interval;
        double starAge = timeInUnitTimes - P[i].StarBirthTime / All.UnitTime_in_s;
        //printf("star %d age: %g\n", i, starAge);
        
        if(starAge > P[i].StarLifeTime / All.UnitTime_in_s)
        {
            ExplodeStar(i);
        }
    } // end of loop over all star particles
}

void CommunicateSNCountsAmongAllTasks()
{
  int counts[NTask];
  for(int i = 0; i < NTask; i++){
      allTasksCounts[i] = 0;
  }
  
  for(int i = 0; i < NTask; i++){
      if(i == ThisTask) counts[i] = N_explosions;
      else counts[i] = 0;
  }
  
  MPI_Allreduce(&counts, &allTasksCounts, NTask, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}

void SetAllGasCellsActive()
{
  for(int i = 0; i < NumGas; i++)
    {
      TimeBinsHydro.ActiveParticleList[i] = i;
    }
  TimeBinsHydro.NActiveParticles = NumGas;
}

void CommunicateSNPositionsAndStarParticleIndicesAmongAllTasks(){
  for(int i = 0; i < maxSNCountPerTimestep; i++){
      for(int j=0; j<3; j++){
          allTasks_SN_positions[i][j] = 0;
      }
      allTasks_SNStarParticleIndices[i] = 0;
  }

  // communicate SN positions among all tasks
  float SN_positions[allTasks_N_explosions][3];
  int SNStarParticleIndices[allTasks_N_explosions];

  int thisTaskFirstSNIndex = 0;
  for(int i = 0; i < ThisTask; i++){
      thisTaskFirstSNIndex += allTasksCounts[i];
  }

  // add thisTaskSNPositions to correct positions in SN_positions array
  for(int i = 0; i < allTasks_N_explosions; i++){
      for(int j=0; j<3; j++){
          if(i >= thisTaskFirstSNIndex && i < thisTaskFirstSNIndex + N_explosions) SN_positions[i][j] = thisTaskSNPositions[i - thisTaskFirstSNIndex][j];
          else SN_positions[i][j] = 0;
      }
      if(i >= thisTaskFirstSNIndex && i < thisTaskFirstSNIndex + N_explosions) SNStarParticleIndices[i] = thisTaskSNStarParticleIndices[i - thisTaskFirstSNIndex];
      else SNStarParticleIndices[i] = 0;
  }

  MPI_Allreduce(&SN_positions, &allTasks_SN_positions, allTasks_N_explosions * 3, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&SNStarParticleIndices, &allTasks_SNStarParticleIndices, allTasks_N_explosions, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}

void FindClosestCellsOnThisTask(int currentSNIndex){
  for(int i = 0; i < amountofcellstodepositenergy; i++)
    {
        closestDistances[i] = 1e20;
    }

    // find closest cells on each task and store indices in closestCellIndices and distances in closestDistances
        for(int i = 0; i < NumGas; i++)
        {
            // calculate distance from SN position to each gas cell
            float squareDistToCell = 0;
            for(int j = 0; j < 3; j++)
            {
              float dist = P[i].Pos[j] - allTasks_SN_positions[currentSNIndex][j];
              
              if(dist > (All.BoxSize/2)) dist = All.BoxSize - dist;

              squareDistToCell += dist * dist;
            }
          
            // just to skip all cells that are very far away:
            if(squareDistToCell > closestDistances[0]) continue; // closestDistances[0] is the furthest away
                
            for(int j = amountofcellstodepositenergy - 1; j >= 0 ; j--) 
            {

                if(squareDistToCell < closestDistances[j])
                {
                    // move all greater distances one step through the array
                    for(int k = 1; k <= j; k++){
                        // j == 0 means this is further away than the (10) closest cells --> doesn't need to be saved anymore
                        closestDistances[k - 1] = closestDistances[k];
                        closestCellIndices[k - 1] = closestCellIndices[k];
                    }
                    // insert new distance and cellindex into the arrays
                    closestDistances[j] = squareDistToCell;
                    closestCellIndices[j] = i;
                    break;
                }
            }
        }

        
        //printf("Task %d: clocestCellIDs and distances-.-\n", ThisTask);
        //for(int i = 0; i < amountofcellstodepositenergy; i++){
        //  if(closestCellIndices[i] == -1) printf("Task (%d): closestCellIndices[%d]: not on this Task\n", ThisTask, i);
        //  else printf("Task (%d): closestCellIndices[%d]: cellID: %d Distance: %g, position: x = %g, y = %g, z = %g\n", ThisTask, i, P[closestCellIndices[i]].ID, closestDistances[i], P[closestCellIndices[i]].Pos[0] , P[closestCellIndices[i]].Pos[1] , P[closestCellIndices[i]].Pos[2]);
        //}
}

void CommunicateClosestCellsOfAllTasks(){
    // communicate closest cells of all tasks
        // create array with all closest distances of all tasks
        int allTasksAmountOfCellsToConsider = amountofcellstodepositenergy * NTask;
        // new array which will have only zeros and the closestDistances array at a position so that these arrays from all tasks can be added together to have an array with all distances of all tasks
        float thisTask_ClosestDistances[allTasksAmountOfCellsToConsider];

        // put in the closestDistances at the right position
        for(int i = 0; i < allTasksAmountOfCellsToConsider; i++){
            if(i >= ThisTask * amountofcellstodepositenergy && i < ThisTask * amountofcellstodepositenergy + amountofcellstodepositenergy)
            { 
                thisTask_ClosestDistances[i] = closestDistances[i - ThisTask * amountofcellstodepositenergy]; // always starts at closestDistances[0] when the condition is met
            } else
            {
                thisTask_ClosestDistances[i] = 0;
            }
        }

        // add up all arrays from all tasks --> in allTasks_ClosestDistances there are all the distances now
        MPI_Allreduce(&thisTask_ClosestDistances, &allTasks_ClosestDistances, allTasksAmountOfCellsToConsider, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

}

void FindAllTaskClosestCellsOnThisTask(){
    // compute all tasks closest cells (compare all entries of allTasks_ClosestDistances with this tasks closestDistances, for the indices a -1 is inserted if a distance from another task is shorter)
        for(int i = 0; i < amountofcellstodepositenergy * NTask; i++)
        {
            // gets stuff wrong if 2 distances are the same
            // if( (i - (i % amountofcellstodepositenergy)) / amountofcellstodepositenergy == ThisTask) continue; // don't need to check the distances we got from this task (also important to not count them double)
            float squareDistToCell = allTasks_ClosestDistances[i];
        
            for(int j = amountofcellstodepositenergy - 1; j >= 0 ; j--)
            {
                if(squareDistToCell < closestDistances[j]) // compare all closest distances of all tasks with the closest distances of this task
                {
                    // move old values down in the array if new value is smaller
                    for(int k = 1; k <= j; k++){
                        // k == 0 means this is further away than the (10) closest cells --> doesn't need to be saved anymore
                        closestDistances[k - 1] = closestDistances[k];
                        closestCellIndices[k - 1] = closestCellIndices[k];
                       
                    }
                    closestDistances[j] = squareDistToCell;
                    closestCellIndices[j] = -1; // index set to -1 if cell is not cell of this task
                    break;
                }
            }
        }
}

int GetSNAmount(int starParticleIndex){
  double numberOfSNe = P[starParticleIndex].Mass * All.UnitMass_in_g / sunMass_in_g * snePerMSun;
  return (int)numberOfSNe;
}

void AddMomentumToCell(int i, int starParticleIndex){
  int numberOfSNe = GetSNAmount(starParticleIndex);

  MyFloat cellDensity = SphP[i].Density;
  double snMomentumToInput_in_msun_km_per_s = pow(10, -0.1 * cellDensity * cellDensity + 0.245 * cellDensity + 6.1); // from fitted function of cluster simulations

  double totalMomentumToAddInCodeUnits = snMomentumToInput_in_msun_km_per_s * sunMass_in_g / All.UnitMass_in_g * 1000 / All.UnitVelocity_in_cm_per_s;
  double momentumToAddInCodeUnits = totalMomentumToAddInCodeUnits / amountofcellstodepositenergy;

  momentumToAddInCodeUnits *= numberOfSNe;

  double velocityToAddInCodeUnits = momentumToAddInCodeUnits / P[i].Mass;

  int dir[3];
  for (int j=0;j<3;j++){
    dir[j] = P[i].Pos[j] - P[starParticleIndex].Pos[j];
  }
  float magnitude = 0;
  for (int j = 0; j < 3; j++) {
      magnitude += dir[j] * dir[j];
  }
  magnitude = sqrt(magnitude);

  for (int j = 0; j < 3; j++) {
      dir[j] = dir[j] / magnitude;
      P[i].Vel[j] += dir[j] * velocityToAddInCodeUnits;
  }

}

void AddEnergyToCell(int i, int starParticleIndex){
  int numberOfSNe = GetSNAmount(starParticleIndex);

  double totalEnergyToAddInCodeUnits = snEnergyToInputInErgs / All.UnitEnergy_in_cgs / P[i].Mass;
  double energyToAddInCodeUnits = totalEnergyToAddInCodeUnits / amountofcellstodepositenergy;

  energyToAddInCodeUnits *= numberOfSNe;
  P[i].energyAddedBySN += All.cf_atime * All.cf_atime * energyToAddInCodeUnits * P[i].Mass;
  P[i].uthermAddedBySN += energyToAddInCodeUnits;
  SphP[i].Utherm += energyToAddInCodeUnits;
  SphP[i].Energy += All.cf_atime * All.cf_atime * energyToAddInCodeUnits * P[i].Mass;

  set_pressure_of_cell(i);
}

void AddMassToCell(int i, int starParticleIndex){
  int numberOfSNe = GetSNAmount(starParticleIndex);
  MyDouble totalMassEjected = numberOfSNe * massEjectedPerSN_in_g / All.UnitMass_in_g;
  MyDouble massToAddToThisCell = totalMassEjected / amountofcellstodepositenergy;
  SphP[i].OldMass += massToAddToThisCell;
  P[starParticleIndex].Mass -= massToAddToThisCell;
}

void AddEnergyAndMassToAllTasksClosestCells(int currentSNIndex){

    int starParticleIndex = allTasks_SNStarParticleIndices[currentSNIndex];

    for(int i = 0; i < amountofcellstodepositenergy; i++){
            
            // skip cells that are not on this task
            if(closestCellIndices[i] == -1) continue;
              
            int closestCellIndex = closestCellIndices[i];

#ifdef MOMENTUM_SN
            AddMomentumToCell(closestCellIndex, starParticleIndex);
#else
            AddEnergyToCell(closestCellIndex, starParticleIndex);
#endif
            AddMassToCell(closestCellIndex, starParticleIndex);
        }
}

#pragma endregion





#ifdef SN_TIMESTEPS

// put this in new function after synchronization (look at Config option FORCE_EQUAL_TIMESTEPS) MPI_Allreduce() timestep.c ?
void adjust_timestep_of_all_particles_to_next_sn()
{
  TIMER_START(CPU_TIMELINE);

  int nextSNBin = get_bin_timestep_to_next_sn_time();

  for(int i = 0; i < NumPart; i++)
    {  // here I need to access all particles on all tasks and also wait for all tasks to compute the nextSNBin before timebins get
       // changed
      if(P[i].TimeBinHydro > nextSNBin)
        {
          P[i].TimeBinHydro = nextSNBin;
        }
    }

  TIMER_STOP(CPU_TIMELINE);
}

int get_bin_timestep_to_next_sn_time()
{  // get timesteps converging to nextSNTime
  for(int bin = 23; bin >= 0; bin--)
    {
      if(test_if_timestep_is_too_large(bin) == 0)
        {
          return bin;
        }
    }
  return 0;  // all timesteps too large --> use minTimestep --> then the SN happens
}

/*! \brief Checks if timestep according to its present timebin is too large
 *
 *  \param[in] bin Timebin to compare to.
 *
 *  \return 0: not too large; 1: too large.
 */
int test_if_timestep_is_too_large(int bin)  // int test_if_grav_timestep_is_too_large(int p, int bin)
{
  int ti_step_bin = bin ? (((int)1) << bin) : 0;

  int ti_step = get_next_sn_time();

  if(ti_step < ti_step_bin)
    return 1;
  else
    return 0;
}

int get_next_sn_time()
{                          // time until next SN
  float nextSNTime = 1e8;  // actually only has to be bigger than the maxTimestep

  for(int i = NumGas; i < NumPart; i++)  // skip gas particles
    {
      if(P[i].StarExploded == 1)
        continue;  // skip already exploded stars (if no more stars are left on this tast return 1e8 --> doesn't change timestep)

      float timeInUnitTimes = All.Ti_Current * All.Timebase_interval;
      float starAge         = timeInUnitTimes - P[i].StarBirthTime / All.UnitTime_in_s;
      float thisStarSNTime  = P[i].StarLifeTime / All.UnitTime_in_s - starAge;

      if(thisStarSNTime < nextSNTime)
        {
          nextSNTime = thisStarSNTime;
        }
    }

  int nextSNTimeInIntegertime = (int)(nextSNTime / All.Timebase_interval);  // convert to integertime

  return nextSNTimeInIntegertime;
}

#endif  // SN_TIMESTEPS

#endif  // XENO_SN