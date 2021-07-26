import random
import math
import csv
import time
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
from functools import partial


def noise_func(noise,dim):
    csv_path=f"./data/train_data{dim}_noise{noise}.csv"
    f = open(csv_path, "w+")
    return csv_path

def simdata(diam,sd,sphere_num,numexpr,noise,csv_path,dim=256):
    """Creates the simulated k-space data

    Args:
        diam : float
            The diameter of the spheres

        sd: float
            The standard deviation as a fraction of the mean size 

        sphere_num: int
            The number of spheres simumated per simulated experiment(If -1 is given
            a random number is chosen each time)

        numexpr: int
            The number of experiments to be simulated for this
            specific parameter combination

        noise:float
            The Gaussian noise to be added in the simualted signal as a fraction of
            the signal intensity in the middle of k-space

        dim: integer
            The size of k-space data (default is 256)
            
        csv_path: string
            Path of csv file where data are stored

    Returns:
        A Boolean value =1 if sucesfull

    """
    start = time.time()
    rad=diam/2
    sd_perc=sd
    sd = sd *diam
    #with open('data/D='+str(diam*100)+'_SD='+str('%0.2f' % (sd*100))+'.csv', "w",newline='') as file:
    with open(csv_path, "a",newline='') as file:
    #with open('data/test_data.csv', "a",newline='') as file:
        csv.register_dialect('myDialect',delimiter = ',')
        diam=round(diam,2)
        sd=round(sd,3)
        info=[diam,sd_perc]
        writer = csv.writer(file,dialect='myDialect')
        #writer.writerow(info)
        
        
        for k in range(numexpr):
            FoV= np.zeros(dim, dtype=np.single)
            x=np.arange((-dim/2),dim/2,1,dtype=np.single)
            if(sphere_num==-1):
                numsph=random.randint(40,4001) #If the user gives -1 as numsph the sphere number changes every time for each profile
            else:
                numsph=sphere_num #else the numsph provided by the user is used for each profile
            for i in range(numsph):
                radius = float(np.random.normal(rad, sd)) #In each sample each sphere is drown from the a normal distribution with the cpecified parameters.
                radius_sq=radius**2
                projection1d=3.14*(radius_sq-np.power(x, 2))
                projection1d=np.where(projection1d<=0,0,projection1d)
                shift=random.randint(-dim/2,dim/2)
                nbubble=np.roll(projection1d, shift)
                FoV=FoV+nbubble
        
            ft = np.fft.fft(FoV)  
            freq = np.fft.fftfreq(FoV.shape[-1])  
            #plt.plot(freq, ft.real, freq, ft.imag)  
            #plt.show()
            freq0=np.fft.fftshift(freq)   
            ft0=np.fft.fftshift(ft) 
            #magn=np.absolute(ft0)
            
            noisere=np.random.randn(ft0.size)*(max(ft0)*noise) #The noise in real and imaginery part should
            noiseim=np.random.randn(ft0.size)*(max(ft0)*noise) #be uncorellated but follow the same distribution
            ftns=ft0.real+noisere+(ft0.imag+noiseim)*1j
            magnns=np.absolute(ftns)
            max_value = np.amax(magnns) 
            magnns = magnns/math.sqrt(max_value) #Scale data with sqrt of intensity in the center of k-space
            #plt.plot(freq0, magnns) 
            #plt.yscale('log') 
            #plt.gca().xaxis.tick_bottom()
            #plt.show()
            #magn.tolist()
            output=np.append(magnns,info)
            writer.writerow(output)
            
           
        file.close()
        end = time.time()
        print("Total runtime was:",end - start)
    return 1


    
def add_header(numexpr,dim,csv_path):
    
    """Adds headers to the csv

    Args:
        numexpr: int
            The number of experiments to be simulated for this
            specific parameter combination

        dim: integer
            The size of k-space data (default is 256)
            
        csv_path: string
            Path of csv file where data are stored

    Returns:
        A pandas dataframe with the simulated data and with headers added.
        Also saves dataframe to csv.

    """
    
    df = pd.read_csv(csv_path)
    kpoints= [number-(dim/2) for number in range(1, dim+1, 1)] 
    kpoints=list(map(str, kpoints))
    kpoints=list(map(lambda x: "kpoint"+x, kpoints))
    column_names=kpoints+['mean','sd']
    df.columns = column_names
    
    #Drop positive half of the columns because they are mirror of negative half 
    i=128
    while i<256:
        df=df.drop(df.columns[128], axis=1)
        i+=1
    df.to_csv (csv_path)
    print("Headers added to csv!")
    return df
 

def sim_df_fix_increment(noise,diam_min,diam_max,diam_step,sd_min,sd_max,sd_step,sphere_num,numexpr,dim):
    """This functions wraps the data simulation and dataframe header addition

    Args:
        noise:float
            The Gaussian noise to be added in the simualted signal as a fraction of
            the signal intensity in the middle of k-space
            
        diam_min : float
            The minimum diameter of the spheres
        diam_max : float
            The maximum diameter of the spheres
        diam_step : float
            The diameter step size of the spheres
        sd_min: float
            The minimum standard deviation as a fraction of the mean size 
        sd_max: float
            The maximum standard deviation as a fraction of the mean size 
        sd_step: float
            The standard deviation step size as a fraction of the mean size 

        sphere_num: int
            The number of spheres simumated per simulated experiment(If -1 is given
            a random number is chosen each time)

        numexpr: int
            The number of experiments to be simulated for this
            specific parameter combination

        dim: integer
            The size of k-space data (default is 256)
          

    Returns:
        A pandas dataframe with the simulated data and with headers added.

    """
    csv_path=f"./data/train_data{numexpr}_noise{noise}.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)
        
    sd=round(sd_min,3)
    while sd  <= sd_max:    
        diam=round(diam_min,2)
        while diam <= diam_max: 
           #print(f"SD parameter {sd} DIAM parameter {diam}")
            simdata(diam,sd,sphere_num,numexpr,noise,csv_path,dim)
            diam +=diam_step
            diam=round(diam,1)
            
        sd += sd_step
        sd=round(sd,3)
        
    df=add_header(numexpr,dim,csv_path)#After completeing the simulation we add headers
    #return df
    

def sim_df_rand_sizes(noise,diam_min,diam_max,sd_min,sd_max,sphere_num,numexpr,dim):
    """This functions wraps the data simulation and dataframe header addition

    Args:
        noise:float
            The Gaussian noise to be added in the simualted signal as a fraction of
            the signal intensity in the middle of k-space
            
        diam_min : float
            The minimum diameter of the spheres
        diam_max : float
            The maximum diameter of the spheres
        diam_numof : integer
            The number of random mean diameters to be simulated between min and max values
        sd_min: float
            The minimum standard deviation as a fraction of the mean size 
        sd_max: float
            The maximum standard deviation as a fraction of the mean size 
        sd_numof: integer
            The number of random standard deviations to be simulated between min and max values
        sphere_num: int
            The number of spheres simumated per simulated experiment(If -1 is given
            a random number is chosen each time)

        numexpr: int
            The number of experiments to be simulated for this
            specific parameter combination

        dim: integer
            The size of k-space data (default is 256)
          

    Returns:
        A pandas dataframe with the simulated data and with headers added.

    """
    csv_path=f"./data/train_data{numexpr}_noise{noise}.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)
        
    
    #Generate array of random mean diameters within min and max values to be used for the simulation
    rnd_diams=np.random.uniform(low=round(diam_min,2), high=round(diam_max,2), size=numexpr)
    
    #Generate array of random standard deviations within min and max values to be used for the simulation
    rnd_sds=np.random.uniform(low=round(sd_min,3), high=round(sd_max,3), size=numexpr)
    
    #Combine random parameters to a list of tuple and loop through it
    rnd_params=list(zip(rnd_diams,rnd_sds))
    
    #Set numexpr to 1 because in this version we have unique random combinations not set classes with multiple experiments
    numexpr=1
    
    for diam,sd in rnd_params:
        simdata(diam,sd,sphere_num,numexpr,noise,csv_path,dim)
    
    df=add_header(numexpr,dim,csv_path)#After completeing the simulation we add headers
    #return df

    
def real_space_profile(diam,sd,sphere_num,dim=256):
    """This functions creates a 1D real space profile of specified sphere(s)

    Args:
        diam : float
            Mean sphere diameter
        sd: float
            Standard deviation of diamteres as a fraction of mean diameter 
        sphere_num: int
            The number of spheres simumated per simulated experiment(If -1 is given
            a random number is chosen each time)
          
    Returns:
        A pandas dataframe with the simulated data and with headers.

    """

    rad=diam/2
    sd_perc=sd
    sd = sd *diam

    FoV= np.zeros(dim, dtype=np.single)
    x=np.arange((-dim/2),dim/2,1,dtype=np.single)
    for i in range(sphere_num):
        radius = float(np.random.normal(rad, sd)) #In each sample each sphere is drown from the a normal distribution with the cpecified parameters.
        radius_sq=radius**2
        projection1d=3.14*(radius_sq-np.power(x, 2))
        projection1d=np.where(projection1d<=0,0,projection1d)
        shift=random.randint(-dim/2,dim/2)
        nbubble=np.roll(projection1d, shift)
        FoV=FoV+nbubble
    x=np.arange(0,dim,1,dtype=np.single)
    df=pd.DataFrame({'signal':FoV,'space':x})
    return df
