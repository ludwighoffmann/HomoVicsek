#Homochirality in Vicsek model with birth-and-death process
#Ludwig A. Hoffmann
#October 07, 2024

#Script to perform some of the analysis of the simulation results.

import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os


#Choose some parameters for which the data should be analyzed
Total_Number_Runs = 1000
Noise_Values = ["001","03","06","1"]
Speed_Values = ["04","08"]
RateRatio_Values = ["2","5","10"]

#Create empty arrays
Length_Run = np.zeros((len(Noise_Values),len(Speed_Values),len(RateRatio_Values),Total_Number_Runs))
Mean_Length_Run = np.zeros((len(Noise_Values),len(Speed_Values),len(RateRatio_Values)))
Median_Length_Run = np.zeros((len(Noise_Values),len(Speed_Values),len(RateRatio_Values)))
STD_Length_Run = np.zeros((len(Noise_Values),len(Speed_Values),len(RateRatio_Values)))

f = open("Length_Run_Statistics.csv","w")
writer = csv.writer(f)

#Loop over all parameters. Computes the mean, median and STD of the length of the run and the evolution of the order parameter.
for i in range(len(Noise_Values)):
    for j in range(len(Speed_Values)):
        for k in range(len(RateRatio_Values)):
            
            print(i)
            print(j)
            print(k)
            print("---")          
                          
            count_state = np.zeros(2)
            order_para_each_run = np.zeros(Total_Number_Runs)
            order_para_rerun_evolution = np.zeros(Total_Number_Runs)
            Last_number_arrays = np.zeros(25) #take random number of boxes over which we average to see which homochiral state we're in
                             
            path = 'Eta_' + str(Noise_Values[i]) + '/Speed_' + str(Speed_Values[j]) + '/RateRatio_' + str(RateRatio_Values[k])

            for Run_Iteration in range(Total_Number_Runs):
                
                if(os.stat(path + "/Run_" + str(Run_Iteration) + ".csv").st_size != 0):
                
                    data = pd.read_csv(path + "/Run_" + str(Run_Iteration) + ".csv")

                Length_Run[i,j,k,Run_Iteration] = data.shape[0]
                
                if(data.shape[0] != 0):

                    for l in range(25):
                        ar = data.iloc[-1][l]
                        Last_number_arrays[l] = np.array(ar.split(" "))[-1].replace("]","")

                    Sum_Last_number_arrays = int(Last_number_arrays.sum()) 

                    if(Sum_Last_number_arrays == 0): 
                        count_state[0] += 1 #State is red, increase by one
                        order_para_each_run[Run_Iteration] = 0
                    else:
                        count_state[1] += 1 #State is blue, increase this count by one
                        order_para_each_run[Run_Iteration] = 1
                
                else:
                    order_para_each_run[Run_Iteration] = 0.5

                order_para_rerun_evolution_temp = order_para_each_run
                order_para_rerun_evolution[Run_Iteration] = order_para_rerun_evolution_temp.sum()/(Run_Iteration+1)
        
            print("---------")
                
            Mean_Length_Run[i,j,k] = np.mean(Length_Run[i,j,k,:])
            Median_Length_Run[i,j,k] = np.median(Length_Run[i,j,k,:])
            STD_Length_Run[i,j,k] = np.std(Length_Run[i,j,k,:])

            writer.writerow(order_para_rerun_evolution[:])
            writer.writerow(Length_Run[i,j,k,:])