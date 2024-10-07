#Homochirality in Vicsek model with birth-and-death process
#Ludwig A. Hoffmann
#October 07, 2024

#Code to run the simulation.

#################################################################################################################################################
#                                                                                                                                               #
#                                                                       Libraries                                                               #
#                                                                                                                                               #
#################################################################################################################################################

import numpy as np
import math
import random
import scipy as sp
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import csv
import os
import multiprocessing as mp



#################################################################################################################################################
#                                                                                                                                               #
#                                                                   Model Parameters                                                            #
#                                                                                                                                               #
#################################################################################################################################################

On_Off_Gillespie = 1 #0                         # Binary Flag to turn Gillespie algorithm on (1) or off (0)
Steps_Save_Image = 9990                         # Every how many iterations is a configuration exported as a figure.
Total_Number_Runs = 50                          # Total number of runs
Steps_Write_csv = 50                            # Every how many iterations is the data added to the csv file.

# Reaction rates for the four different reactions we have in the Gillespie algorithm. Here X = L, R the left and right chiral particles and A is the food.
K = 100                                         # Cell division consuming food: X+A -> 2X
kd = 10                                         # Cell death (cell X decays to food A): X -> A
ks = 0.00                                       # Switch of orientation of cell: L <-> R    
kn = 0.00                                       # Food "spontaneously" becoming cell: A -> X

# How often many steps does the Gillespie algorithm run (steps) and how many times is the experiment repeated (cycles)
steps = 10                                      # Number of reactions per trajectory
cycles = 1                                      # Number of trajectories iterated over. [Has to be 1 for the code below. To have more cycles the code has to be modified.] 

# Model parameters for the Viscek model
L = 20.0                                        # Length of side of the system
rho = 2.0                                       # Density of particles
Length_Box = 2.                                 # Length of a box that (kind of) sets length scale of reaction that is the box in which we evaluate the Gillespie algorithm
r0 = Length_Box/2                               # Length of interaction radius for alignment interaction in Viscek model
deltat = 1.0                                    # Time step used in Viscek model
factor = 0.4                                    # (Arbitrary) prefactor (just has to be <1) used in velocity of Viscek model 
v0 = factor                                     # Velocity of Viscek model
time_steps_viscek = 50000                       # Number of iterations for which the Viscek algorithm is running
eta = 1                                         # Magnitude of the noise in the orientation dynamics
omega = 0.00                                    # Magnitude of the deterministic rotation term in the orientation dynamics

# Initial number of left and right particles as well as number of food
N_Particle = 2                                  # Total initial number of particles
N0_Left = int(N_Particle/2)                     # Total initial number of left particles
N0_Right = N_Particle - N0_Left                 # Total initial number of right particles
N_Food = int(rho*L**2)                          # Total initial number of food particles

# Initialize random seed and plot
#np.random.seed()                               # Fix random seed for repeatability
fig, ax = plt.subplots(figsize=(12,12))         # Size of figure
ax.axis([0,L,0,L])                              # Set fixed axis for figure



#################################################################################################################################################
#                                                                                                                                               #
#                                                                       Functions                                                               #
#                                                                                                                                               #
#################################################################################################################################################

def weighted_choice(weights): 
    """Define a function for weighting required for the Gillespie algorithm."""
    #Adapted from https://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python#id4. 
    
    totals = []
    running_total = 0

    for w in weights:
        running_total += w
        totals.append(running_total)

    rnd = random.random() * running_total
    for i, total in enumerate(totals):
        if rnd < total:
            return i
def Gillespie_Algorithm(N_L0, N_A0, N_R0):
    """Main code for Gillespie algorithm with N_R0, N_L0, N_A0 the initial number of right, left, food particles that are reacting"""
    #Adapted from https://lewiscoleblog.com/gillespie-algorithm.
    
    # Set up arrays for reaction time and particles of size cycles x step, so that for each cycle the reaction runs steps times.
    T = np.zeros((cycles, steps+1))
    N_L = np.zeros((cycles, steps+1))
    N_A = np.zeros((cycles, steps+1))
    N_R = np.zeros((cycles, steps+1))

    # Store initial conditions in the arrays
    N_L[:,0] = N_L0
    N_A[:,0] = N_A0
    N_R[:,0] = N_R0

    # For each cycle loop through all the steps of the reactions
    for i in range(cycles):
        for j in range(steps):
            
            # Calculate the updated overall reaction rate R from the number of particles present and the fixed reaction rates.
            R = K * N_A[i,j] * (N_R[i,j] + N_L[i,j]) + kd * (N_R[i,j] + N_L[i,j]) + ks * (N_R[i,j] + N_L[i,j]) + 2 * kn * N_A[i,j]      

            # Calculate time to next reaction according to the paper by Gillespie. Then store the reaction time in the array T
            u1 = np.random.random()
            tau = 1/R * np.log(1/u1)
            T[i, j+1] = T[i,j] + tau

            # Define for each reaction a certain probability with which it occurs. Then select which reaction occurs using the weighted_choice function.
            P1 = K * N_A[i,j] * N_R[i,j] / R
            P2 = K * N_A[i,j] * N_L[i,j] / R
            P3 = kd * N_R[i,j] / R
            P4 = kd * N_L[i,j] / R
            P5 = ks * N_R[i,j] / R
            P6 = ks * N_L[i,j] / R
            P7 = kn * N_A[i,j] / R
            P8 = kn * N_A[i,j] / R

            choice_reaction = weighted_choice([P1,P2,P3,P4,P5,P6,P7,P8])
            
            # Update populations depending on which reaction occured
            if choice_reaction == 0:
                N_A[i,j+1] = N_A[i,j] - 1
                N_R[i,j+1] = N_R[i,j] + 1
                N_L[i,j+1] = N_L[i,j]    
            elif choice_reaction == 1:
                N_A[i,j+1] = N_A[i,j] - 1
                N_R[i,j+1] = N_R[i,j]
                N_L[i,j+1] = N_L[i,j] + 1   
            elif choice_reaction == 2:
                N_A[i,j+1] = N_A[i,j] + 1
                N_R[i,j+1] = N_R[i,j] - 1
                N_L[i,j+1] = N_L[i,j]
            elif choice_reaction == 3:
                N_A[i,j+1] = N_A[i,j] + 1
                N_R[i,j+1] = N_R[i,j]
                N_L[i,j+1] = N_L[i,j] - 1   
            elif choice_reaction == 4:
                N_A[i,j+1] = N_A[i,j]
                N_R[i,j+1] = N_R[i,j] - 1
                N_L[i,j+1] = N_L[i,j] + 1  
            elif choice_reaction == 5:
                N_A[i,j+1] = N_A[i,j]
                N_R[i,j+1] = N_R[i,j] + 1
                N_L[i,j+1] = N_L[i,j] - 1  
            elif choice_reaction == 6:
                N_A[i,j+1] = N_A[i,j] - 1
                N_R[i,j+1] = N_R[i,j] + 1
                N_L[i,j+1] = N_L[i,j]
            elif choice_reaction == 7:
                N_A[i,j+1] = N_A[i,j] - 1
                N_R[i,j+1] = N_R[i,j]
                N_L[i,j+1] = N_L[i,j] + 1 
    
    #Now at the end of the reactions return only the value at the last step for one cycle for the number of particles 
    return int(N_L[0,steps]), int(N_A[0,steps]), int(N_R[0,steps])

################################################################################################################################################

def Initilization():
    """Function to initialize the system"""
    
    #np.random.seed()
    
    # Initially we start with N0_Left left particles, N0_Right right particles and N_Food times food. For each we create a position, an orientation, the rotation factor (-1 for left, 0 for food, and 1 for right) as well as a box number (set to zero here and updated to the correct value using the function Set_Box_Number).
    # For the food we set the initial orientation to 0.
    Pos_Particle_x_Left = np.random.uniform(0,L,size=N0_Left)
    Pos_Particle_y_Left = np.random.uniform(0,L,size=N0_Left)
    Orient_Particle_Left = np.random.uniform(-np.pi, np.pi,size=N0_Left)
    Rotation_Factor_Left = [-1] * N0_Left
    Box_Number_Left = [0] * N0_Left
    
    Pos_Particle_x_Right = np.random.uniform(0,L,size=N0_Right)
    Pos_Particle_y_Right = np.random.uniform(0,L,size=N0_Right)
    Orient_Particle_Right = np.random.uniform(-np.pi, np.pi,size=N0_Right)
    Rotation_Factor_Right = [1] * N0_Right
    Box_Number_Right = [0] * N0_Right
    
    Pos_Food_x = np.random.uniform(0,L,size=N_Food)
    Pos_Food_y = np.random.uniform(0,L,size=N_Food)
    Orient_Food = np.random.uniform(-np.pi, np.pi,size=N_Food)
    Rotation_Factor_Food = [0] * N_Food
    Box_Number_Food = [0] * N_Food

    # For each left, right, and food create an array and then concatenate them into one array such that we have an array of all the particles in the system (called Total_Array) and we distinguish them by the rotation factor in the zeroth column.
    Left_Particle_Array = np.array([Rotation_Factor_Left,Box_Number_Left,Pos_Particle_x_Left,Pos_Particle_y_Left,Orient_Particle_Left]).T
    Right_Particle_Array = np.array([Rotation_Factor_Right,Box_Number_Right,Pos_Particle_x_Right,Pos_Particle_y_Right,Orient_Particle_Right]).T
    Food_Array = np.array([Rotation_Factor_Food,Box_Number_Food,Pos_Food_x,Pos_Food_y,Orient_Food]).T
    TotalArray = np.concatenate((Left_Particle_Array,Right_Particle_Array,Food_Array))
    
    # For the plot of the inital configuration we create an array of only the particles (Particle_Array) and then make an quiver plot for the particles and a scatter plot for the food.
    # Note that the color for the particles here is just black, independent of their chirality.
    # Commented out because of the way the code is written at the moment we save a figure only from the Update_Particle_Pos_Orient function
    
    #Particle_Array = np.concatenate((Left_Particle_Array,Right_Particle_Array))
    #vector_particle = ax.quiver(Particle_Array[:,2], Particle_Array[:,3], np.cos(Particle_Array[:,4]), np.sin(Particle_Array[:,4]))
    #scatter_food = ax.scatter(Food_Array[:,2], Food_Array[:,3],s=10, color = "g")
    
    return TotalArray#,vector_particle,scatter_food

def Set_Box_Number(TotalArray):
    """Function to set the correct box number of each particle"""
    
    # To do this take a box of length L and divide the syste into (L/Length_Box)**2 boxes, numbered from top left to bottom right like the following example for 4x4 boxes:
    #
    #           0   1   2   3
    #           4   5   6   7
    #           8   9   10  11
    #           12  13  14  15
    

    for i in range(len(TotalArray)):
        TotalArray[i][1] = int(TotalArray[i][2]/Length_Box) + L/Length_Box * int(TotalArray[i][3]/Length_Box)
    
    return TotalArray
def Update_Particle_Pos_Orient(TotalArray,Iteration,Run_Counter,Break_Flag):
    """Function to update the position and the orientation of every particle according to the Viscek model and plot the configuration before the update"""
    
    
    # We need to seperate the particles from the food, thus create two arrays, one for the particles and one for the food, and fill them in a loop depending on wether rotation factor is zero or non-zero.
    particle_information = []
    food_information = []
    
    for i in range(len(TotalArray)):
        if(TotalArray[i][0] != 0):
            particle_information.append(TotalArray[i,:])
        elif(TotalArray[i][0] == 0):
            food_information.append(TotalArray[i,:])
            
    particle_information = np.array(particle_information)
    food_information = np.array(food_information)
    
    # Extract data from these arrays and define a few variables (orienation, position, rotation factor of particles and position of food) to make expressions shorter below. If there is no food anymore make an empty array for it.
    orient_particle = particle_information[:,4]
    pos_particle = np.vstack((particle_information[:,2], particle_information[:,3])).T
    rotation_particle = particle_information[:,0]
    if(len(food_information) != 0):
        pos_food = np.vstack((food_information[:,2], food_information[:,3])).T
    else:
        pos_food = []
    
    # Create a color array where it is red for left particles (if the rotation factor is -1) and blue for right particles.
    # Count how many red and left particles in total and break if no left or right particles
    ColorArray = []
    Tot_num_r = 0
    Tot_num_b = 0
    
    for i in range(len(particle_information)):
        if(particle_information[i,0] == -1):
            ColorArray.append("r")
            Tot_num_r += 1
        elif(particle_information[i,0] == 1):
            ColorArray.append("b")
            Tot_num_b += 1
    
    if(Tot_num_r == 0 or Tot_num_b == 0):
        
        fig, ax = plt.subplots(figsize=(12,12))
        ax.axis([0,L,0,L])
        vector_particle = ax.quiver(particle_information[:,2], particle_information[:,3], np.cos(particle_information[:,4]), np.sin(particle_information[:,4]), color = ColorArray)
        if(len(food_information) != 0):
            scatter_food = ax.scatter(pos_food[:,0], pos_food[:,1],s=10, color = "g")
    
        plt.savefig("Output/" + "image" + str(Iteration)+ "_Run_" + str(Run_Counter) + ".jpg")
        
        plt.close()
        
        Break_Flag = True
    
    # Create a figure of the configuration every x iterations and save it to the folder Output. Only plot the food if there is any.
    if(Iteration%Steps_Save_Image == 0):
        fig, ax = plt.subplots(figsize=(12,12))
        ax.axis([0,L,0,L])
    
        vector_particle = ax.quiver(particle_information[:,2], particle_information[:,3], np.cos(particle_information[:,4]), np.sin(particle_information[:,4]), color = ColorArray)
        if(len(food_information) != 0):
            scatter_food = ax.scatter(pos_food[:,0], pos_food[:,1],s=10, color = "g")
    
        plt.savefig("Output/" + "image" + str(Iteration)+ "_Run_" + str(Run_Counter) + ".jpg")
        
        plt.close()
    
    
    # Now update the configuration according to the Viscek model. The following code is based on the implementation in https://francescoturci.net/2020/06/19/minimal-vicsek-model-in-python/.
    
    # First create a tree of all the particles in the system and compute the distance between all particles that are closer to each other than r0. Then sum over the angle of each close particle to get an expression for the angle in the alignment interaction term.

    tree_particle = cKDTree(pos_particle,boxsize=[L,L])                                                     # Finds next nearest neighbors for every point
    dist = tree_particle.sparse_distance_matrix(tree_particle, max_distance=r0,output_type='coo_matrix')    # Computes a distance matrix between two cKDTrees, leaving as zero any distance greater than max_distance. In every line there is the distance XXX between point x and point y, such that print(dist) has the form: (x,y) XXX. The distance is set to zero if it is bigger than max_distance. For print(dist) the lines with distance > max_distance are not printed. Writing dist as an array then print(dist.toarray()) gives a matrix with entry 0 whenever distance > max_distance.
    data = np.exp(orient_particle[dist.col]*1j)                                                             # dist.col returns the column of all points y such that orient_particle[dist.col] gives an array of orient_particleations of the points y. Note that a particle can appear multiple times in dist.col (if it is closer than max_distance to more than one particle). Then data is an array of complex numbers computed as the exponential of orient_particleation times complex unit.
    neigh = sparse.coo_matrix((data,(dist.row,dist.col)), shape=dist.get_shape())                           # Now construct  a new sparse marix with entries in the same places ij of the dist matrix. This now is a matrix that in row x has entries at point (x,y) that are zero if a point y is further away than max_distance and the angle orient_particle[y] if they are closer than max_distance.
    S = np.squeeze(np.asarray(neigh.tocsr().sum(axis=1)))                                                   # Sum along the columns (sum over j) and make resulting structure into 1d array.

    # Update the orienation of each particle according to (1) the nearby particles (via S), (2) by adding some noise of magnitude eta, and (3) by a deterministing rotation term with magnitude omega and with sign dependend on chirality of particle
    
    orient_particle = np.angle(S)+eta*np.random.uniform(-np.pi, np.pi, size=len(particle_information)) + np.multiply(omega,rotation_particle)
    
    # Update the position of each particle and using/enforcing periodic boundary conditions
 
    cos, sin= np.cos(orient_particle), np.sin(orient_particle)
    pos_particle[:,0] += cos * v0
    pos_particle[:,1] += sin * v0
    pos_particle[pos_particle>L] -= L
    pos_particle[pos_particle<0] += L

    # Update the particle_information array and TotalArray.
    
    particle_information[:,2] = pos_particle[:,0]
    particle_information[:,3] = pos_particle[:,1]
    particle_information[:,4] = orient_particle
    if(len(food_information) != 0):
        TotalArray = np.concatenate((particle_information,food_information))
    else:
        TotalArray = particle_information
    
    return TotalArray,Break_Flag


        
def Counting_Per_Box(TotalArray,writer,Iteration,Break_Flag):
    """Function to  count the box and the number of left, food, right particles present in this box."""
    
    #For every box we want to count the box and the number of left, food, right particles present in this box and store them in the array Numbers_Per_Box in this order. For this create an empty array of dimension (Number of Boxes) x 4 (box number, left, food, right). Then loop through TotalArray and loop through all boxes to look for each entry in TotalArray in which box it is and then add 1 in the array Numbers_Per_Box with the column depending on wether the rotation factor is -1, 0, or +1.
    Numbers_Per_Box = np.zeros([int((L/Length_Box)**2),4], dtype = int) 

    for i in range(len(TotalArray)):
        for j in range(len(Numbers_Per_Box)):
            Numbers_Per_Box[j][0] = j
            if(TotalArray[i][1] == j):
                if(TotalArray[i][0] == -1):
                    Numbers_Per_Box[j][1] += 1
                elif(TotalArray[i][0] == 0):
                    Numbers_Per_Box[j][2] += 1
                elif(TotalArray[i][0] == 1):
                    Numbers_Per_Box[j][3] += 1

    if(Iteration%Steps_Write_csv == 0):
        
        writer.writerow(Numbers_Per_Box)
        
    if Break_Flag:
        
        writer.writerow(Numbers_Per_Box)

    return(Numbers_Per_Box)
 

def Counting_LFR_Box(TotalArray,j,Position_In_Array_Left,Position_In_Array_Food,Position_In_Array_Right): 
    """Count how many left, food, right parrticles there are in a given box j"""
    
    # Add to Position_In_Array_Left all left particles in box j and to Position_In_Array_Food all food particles in box j
    for k in range(len(TotalArray)):                     
        if(TotalArray[k,0] == -1 and int(TotalArray[k,1]) == j):
            Position_In_Array_Left.append(k)
        elif(TotalArray[k,0] == 0 and int(TotalArray[k,1]) == j):
            Position_In_Array_Food.append(k)
        elif(TotalArray[k,0] == 1 and int(TotalArray[k,1]) == j):
            Position_In_Array_Right.append(k)
    return(Position_In_Array_Left,Position_In_Array_Food,Position_In_Array_Right)

def Random_Position(j,One_Destroyed,Two_Destroyed,Three_Destroyed,Numbers_Per_Box,Updated_Numbers_Per_Box,Position_In_Array_Left,Position_In_Array_Food,Position_In_Array_Right,Random_Position_In_Array):
    """Choose random particles to be destroyed."""
    if One_Destroyed == 1:

        Random_Position_In_Array.append(random.sample(Position_In_Array_Left, np.absolute(Updated_Numbers_Per_Box[j,1] - Numbers_Per_Box[j,1]))) # Make a random choice of the left particles where the number of particles we choose is given by how many particles were destroyed. Add these positions to Random_Position_In_Array

    if Two_Destroyed == 1:

        Random_Position_In_Array.append(random.sample(Position_In_Array_Food, np.absolute(Updated_Numbers_Per_Box[j,2] - Numbers_Per_Box[j,2]))) # Same but for food.

    if Three_Destroyed == 1:

        Random_Position_In_Array.append(random.sample(Position_In_Array_Right, np.absolute(Updated_Numbers_Per_Box[j,3] - Numbers_Per_Box[j,3]))) # Same but for right.
    
    Random_Position_In_Array = np.concatenate(Random_Position_In_Array) # Make a 1d array out of Random_Position_In_Array
    
    random.shuffle(Random_Position_In_Array) #Shuffel array
    
    return(Random_Position_In_Array)

def Conversion(j,One_Destroyed,Two_Destroyed,Three_Destroyed,One_Created,Two_Created,Three_Created,Numbers_Per_Box,Updated_Numbers_Per_Box,Random_Position_In_Array,TotalArray): """Convert particles in box according to result of Gillespie algorithm."""
    
    Number_Destroyed_Total = int(One_Destroyed * int(Numbers_Per_Box[j,1] - Updated_Numbers_Per_Box[j,1]) + Two_Destroyed * int(Numbers_Per_Box[j,2] - Updated_Numbers_Per_Box[j,2]) + Three_Destroyed * int(Numbers_Per_Box[j,3] - Updated_Numbers_Per_Box[j,3])) #Count how many particles were destroyed in total
    
    if One_Created + Two_Created + Three_Created == 1: #If only one kind was created find out which one and assign to Created_Rot_Number the rotation value number of the kind that was created
    
        if One_Created == 1:
            
            Created_Rot_Number = -1
        
        elif Two_Created == 1:
            
            Created_Rot_Number = 0
        
        elif Three_Created == 1:
            
            Created_Rot_Number = 1
            
        for m in range(Number_Destroyed_Total): # Now convert the particles into the created particles in the total array by just changing the rotation factor appropriately
            TotalArray[np.array(Random_Position_In_Array[m]).item(),0] = Created_Rot_Number
        
    elif One_Created + Two_Created + Three_Created == 2: #If two kinds were created go through all possible cases and change rotation factor of appropriate number of particles
    
        if(Two_Created == 1 and Three_Created == 1):
    
            for l in range(int(Updated_Numbers_Per_Box[j,2] - Numbers_Per_Box[j,2])):
                TotalArray[np.array(Random_Position_In_Array[l]).item(),0] = 0
            for m in range(int(Updated_Numbers_Per_Box[j,3] - Numbers_Per_Box[j,3])):
                TotalArray[np.array(Random_Position_In_Array[m+int(Updated_Numbers_Per_Box[j,2] - Numbers_Per_Box[j,2])]).item(),0] = 1 
                
        elif(One_Created == 1 and Three_Created == 1):
    
            for l in range(int(Updated_Numbers_Per_Box[j,1] - Numbers_Per_Box[j,1])):
                TotalArray[np.array(Random_Position_In_Array[l]).item(),0] = -1
            for m in range(int(Updated_Numbers_Per_Box[j,3] - Numbers_Per_Box[j,3])):
                TotalArray[np.array(Random_Position_In_Array[m+int(Updated_Numbers_Per_Box[j,1] - Numbers_Per_Box[j,1])]).item(),0] = 1
                
        elif(One_Created == 1 and Two_Created == 1):    
                
            for l in range(int(Updated_Numbers_Per_Box[j,1] - Numbers_Per_Box[j,1])):
                TotalArray[np.array(Random_Position_In_Array[l]).item(),0] = -1
            for m in range(int(Updated_Numbers_Per_Box[j,2] - Numbers_Per_Box[j,2])):
                TotalArray[np.array(Random_Position_In_Array[m+int(Updated_Numbers_Per_Box[j,1] - Numbers_Per_Box[j,1])]).item(),0] = 0
    
    return(TotalArray)
    
def Update_Array(Numbers_Per_Box,TotalArray,i):
    """Function to run the Gillespie algorithm on the system.""" 
    
    #We do this by considering each subsytem box as an isolated system and let the Gillespie algorithm run in each system with the inital values given by the number of particles present before the algorithm is run in every box. From running the Gillespie algorithm we only get the updated number of particles per box back. We decide which particles at which position to convert randomly.
    
    # First create an new array where we update the number of left, right, and food
    
    Updated_Numbers_Per_Box = Numbers_Per_Box.copy()
    
    # Now or each box run the Gillespie algorithm and after that look at how many particles of each type are in each box and update TotalArray accordingly. The Gillespie algorithm does not tell us which particle at which position was converted so to update TotalArray we randomly choose which of the particles to convert.
        
    for j in range(len(Numbers_Per_Box)):
        if(int(Numbers_Per_Box[j,1]+Numbers_Per_Box[j,3]) != 0): # Only run the Gillespie algorithm if there are cells in the box. In particular, don't run if there is only food in the box.
            Updated_Numbers_Per_Box[j,1],Updated_Numbers_Per_Box[j,2],Updated_Numbers_Per_Box[j,3] = Gillespie_Algorithm(Numbers_Per_Box[j,1],Numbers_Per_Box[j,2],Numbers_Per_Box[j,3]) # Update how many particles of each type are in the box from result of Gillespie algorithm.
        if(Updated_Numbers_Per_Box[j,1] != Numbers_Per_Box[j,1] or Updated_Numbers_Per_Box[j,2] != Numbers_Per_Box[j,2] or Updated_Numbers_Per_Box[j,3] != Numbers_Per_Box[j,3]): # Check if anything was changed by the Gillespie algorithm
            
            # Now look at each case seperately. Depending on what kind of particles were created or destroyed update the other particles
            
            Position_In_Array_Left = []                         # Make an array to store where all destroyed particles are, here where the left and food particles are
            Position_In_Array_Food = []                         #
            Position_In_Array_Right = []
            Random_Position_In_Array = []                       # Make an array to store a random selection of positions of destroyed particles
            
            Position_In_Array_Left,Position_In_Array_Food,Position_In_Array_Right = Counting_LFR_Box(TotalArray,j,Position_In_Array_Left,Position_In_Array_Food,Position_In_Array_Right) #make array with position of all L, R, F in the given box
            
            if(Updated_Numbers_Per_Box[j,1] < Numbers_Per_Box[j,1]): #set flags depending on whether each kind destroyed or created
                One_Destroyed,One_Created = 1,0
            elif(Updated_Numbers_Per_Box[j,1] == Numbers_Per_Box[j,1]):
                One_Destroyed,One_Created = 0,0
            elif(Updated_Numbers_Per_Box[j,1] > Numbers_Per_Box[j,1]):
                One_Destroyed,One_Created = 0,1
            
            if(Updated_Numbers_Per_Box[j,2] < Numbers_Per_Box[j,2]):
                Two_Destroyed,Two_Created = 1,0
            elif(Updated_Numbers_Per_Box[j,2] == Numbers_Per_Box[j,2]):
                Two_Destroyed,Two_Created = 0,0
            elif(Updated_Numbers_Per_Box[j,2] > Numbers_Per_Box[j,2]):
                Two_Destroyed,Two_Created = 0,1
                
            if(Updated_Numbers_Per_Box[j,3] < Numbers_Per_Box[j,3]):
                Three_Destroyed,Three_Created = 1,0
            elif(Updated_Numbers_Per_Box[j,3] == Numbers_Per_Box[j,3]):
                Three_Destroyed,Three_Created = 0,0
            elif(Updated_Numbers_Per_Box[j,3] > Numbers_Per_Box[j,3]):
                Three_Destroyed,Three_Created = 0,1
            
            Random_Position_In_Array = Random_Position(j,One_Destroyed,Two_Destroyed,Three_Destroyed,Numbers_Per_Box,Updated_Numbers_Per_Box,Position_In_Array_Left,Position_In_Array_Food,Position_In_Array_Right,Random_Position_In_Array) #pick random particles
 
            TotalArray = Conversion(j,One_Destroyed,Two_Destroyed,Three_Destroyed,One_Created,Two_Created,Three_Created,Numbers_Per_Box,Updated_Numbers_Per_Box,Random_Position_In_Array,TotalArray) #convert random particles according to numbers from Gillespie
    
    return TotalArray

def Dynamics(Number_Iterations,Run_Counter):
    """Function that calls in the right order all the functions needed to couple the Viscek model to the Gillespie algorithm. We let the system run for Number_Iterations steps of the Viscek model."""
    
    f = open("Run_" + str(Run_Counter) + ".csv", 'w')
    writer = csv.writer(f)
    
    np.random.seed()
    
    print(np.random.seed())
    
    TotalArray = Initilization()                                        # Initialize the system and create an array of all the particles
    Break_Flag = False
    for i in range(Number_Iterations):                                  # Run the following Number_Iterations times
    
        if not Break_Flag:
            TotalArray = Set_Box_Number(TotalArray)                         # Set the correct box number for each particle in total array
            TotalArray,Break_Flag = Update_Particle_Pos_Orient(TotalArray,i,Run_Counter,Break_Flag)           # Update the position and orientation of each particle according to the Vicsek model. Before making the update save a figure of the configuration.
            TotalArray = Set_Box_Number(TotalArray)                         # Update again to correct box number of each particle that has changed because of the movement in the Vicsek model.
            Numbers_Per_Box = Counting_Per_Box(TotalArray,writer,i,Break_Flag)                  # Count how many particles are in each box.
                                                                            #
            if(On_Off_Gillespie == 1):                                      # Check if the Gillespie algorithm is turned on    
                TotalArray = Update_Array(Numbers_Per_Box, TotalArray,i)    # Let the Gillespie algorithm run for each of the subset boxes. In the end update TotalArray.
        elif Break_Flag:
            
            break
    
    f.close()
            
#################################################################################################################################################
#                                                                                                                                               #
#                                                                  Running the Program                                                          #
#                                                                                                                                               #
#################################################################################################################################################


n_cores = os.environ['SLURM_JOB_CPUS_PER_NODE']

pool = mp.Pool(processes=int(n_cores))

res = pool.starmap(Dynamics,[(time_steps_viscek,Run_Counter) for Run_Counter in range(Total_Number_Runs)])

pool.close()

pool.join()

#for Run_Counter in range(Total_Number_Runs):
    
    #f = open("Run_" + str(Run_Counter) + ".csv", 'w')
    #writer = csv.writer(f)

    #Dynamics(time_steps_viscek,Run_Counter) #Let the system run for time_steps_viscek steps of the Viscek model. After each of the steps in the Viscek model the Gillespie algorithm runs steps times.

    #f.close()








