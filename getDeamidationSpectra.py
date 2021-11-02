# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:58:06 2021

@author: Bruger
"""
import pandas as pd
import numpy as np

from scipy import stats, optimize
from pyopenms import MSExperiment, MzMLFile, EmpiricalFormula, CoarseIsotopePatternGenerator

def getDeamidationSpectra(file_name,time_int,charge,peak_count,save=False,conf=0.05):        
        
    #Define empty experiment
    exp = MSExperiment()
        
    #load data from file from file in current directory
    MzMLFile().load("./"+file_name+".mzML", exp)
    
    #The emeperical formulars of HI (ins_d0) and deamidated HI (ins_d1)
    ins_d0 = EmpiricalFormula("C257H383N65O77S6")
    ins_d1 = EmpiricalFormula("C257H382N64O78S6")
    
    #Array of isotopic patter of HI and deamidated HI,alligned by appending and prepending [0]
    ins_d0_r = np.array([iso.getIntensity() for iso in ins_d0.getIsotopeDistribution(CoarseIsotopePatternGenerator(peak_count)).getContainer()]+[0])
    ins_d1_r = np.array([0]+[iso.getIntensity() for iso in ins_d1.getIsotopeDistribution(CoarseIsotopePatternGenerator(peak_count)).getContainer()])
    
    #Finds the index of the most abundant peak for normal HI, to use as normailization peak
    norm_bp = np.argmax(ins_d0_r)
    
    #Function to fit, takes d as argument, the fraction of deamidated, and return the combined isotopic pattern
    def FitInsulinWith1Deamidations(d):
        #Combine the isotopic patterns
        result = np.array([d0*(1-d) + d1*d for d0,d1 in zip(ins_d0_r,ins_d1_r)])
        # returns the combined pattern, normalized to 
        return np.vectorize(lambda x: x / result[norm_bp])(result) 
    
    time_int = (time_int[0]*60,time_int[1]*60)
    
    monoWeight = ins_d0.getMonoWeight()

    mz_lim = ((monoWeight/charge)+1-(1/(2*charge)),monoWeight/charge+ 1 + (peak_count/charge)+ 1/(2*charge))
    mz_fil = [(t,[(mz,a) for mz,a in zip(*s.get_peaks()) if mz_lim[0]<= mz <=mz_lim[1]]) for s in exp.getSpectra() if time_int[0]<=(t := s.getRT()) <= time_int[1]]    
                        
    opt = []   
        
    fittingFunction = lambda x,d: FitInsulinWith1Deamidations(d)[x]
    normalize = lambda x: 0 if x == 0 else x / norm_bp_a
        
        
    for t,a_list in mz_fil:
        tic = sum([a for _,a in a_list])
        max_list = np.zeros(peak_count+1)
        
        for a,p in [(a,int((mz - mz_lim[0]) / (1/charge))) for mz,a in a_list]: 
            if max_list[p] < a: max_list[p] = a
                                        
        if max_list[norm_bp] == 0:
            continue

        norm_bp_a = max_list[norm_bp]
        
        tmp_opt=optimize.curve_fit(
            fittingFunction,
            range(peak_count+1),
            np.vectorize(normalize)(max_list),
            p0=(0.5),
            bounds=(0,1))
        
        opt.append((t,tmp_opt[0][0],tmp_opt[1][0][0],tic))
    

    #define dataframe
    df = pd.DataFrame(data=opt, columns=["Time","D1_frac", "s","TIC_abs"])
    #Set column "Time" as index
    df = df.set_index("Time")
    #Convert "Time" from seconds to minutes
    df.index = df.index.map(lambda x: x / 60)        
                
    df["D0_frac"] = df["D1_frac"].apply(lambda x: 1-x)
    df["TIC"] = df["TIC_abs"].apply(lambda x: x/df["TIC_abs"].max())   
    df["D1"] = df["D1_frac"]*df["TIC"]
    df["D0"] = df["D0_frac"]*df["TIC"]
    #np.sqrt(np.diag([x]))
    df["sigma"] = df["s"].apply(lambda x:np.sqrt(x))
    df["sig95"] = df["sigma"]*stats.t.ppf(1-conf/2, peak_count)/np.sqrt(peak_count)   
    
    if save:
        df.to_csv("./"+file_name+".csv",index=True)
    
    return df
