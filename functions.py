import numpy as np
import os
import glob


def time_avg(l):
    n=int(len(l)/2)
    l[1:n]=(l[1:n]+l[-1:-n:-1])/2
    return l[:n+1]


def read_files_mock(folder_name = 'practice_data_2pt',file_name_starting = '2pt_0_*.txt'):

    #file_path = os.path.join(folder_name, file_name)

    # Pattern to match files starting with '2pt_0_'
    pattern = os.path.join(folder_name, file_name_starting)

    # Use glob to find all files matching the pattern
    file_paths = glob.glob(pattern)

    coloumn_5=[]
    # Loop through the matched files and read each one
    for file_path in file_paths:
        try:
            data = np.loadtxt(file_path)  
            coloumn_5.append(data[:,4])
            #print(f"Data from {os.path.basename(file_path)}:")
            #print(data)

        except Exception as e:
            print(f"An error occurred while reading {file_path}: {e}")
    
    return np.array(coloumn_5)


def jackknife_multiple_col(eff_E):
    Nbins=len(eff_E)
    
    total_E_t=np.sum(eff_E,axis=0)
    jacked_bins=(total_E_t-eff_E)/(Nbins-1)
    jackknife_mean=np.mean(jacked_bins,axis=0)

    #jackknife_error=np.sqrt((Nbins-1)/(Nbins))*np.sqrt(np.sum((jacked_bins-jackknife_mean)**2,axis=0))
    jackknife_error=np.sqrt((Nbins-1))*np.std(jacked_bins,axis=0)

    return (jackknife_mean,jackknife_error)


#The following function takes 2pt function but returns energy
def jk_2pt_energy(eff_E):
    Nbins=len(eff_E) #Calculated the no of columns(each row has values for different simulations of the same thing)
    
    #performing the jackknife resampling
    total_E_t=np.sum(eff_E,axis=0)
    sampled_2pt=(total_E_t-eff_E)/(Nbins-1) #resampled 2pt function values

    ''' Mean Energy value calculated by first calculating the mean 2pt functions at different time
    sampled_2pt_mean=np.mean(sampled_2pt,axis=0)
    Energy=np.log(sampled_2pt_mean[:-1]/sampled_2pt_mean[1:])
    '''
    #The 2pt function bin is directly converted into Energy bin, then mean Energy is calculated
    Energy_bins=np.log(sampled_2pt[:,:-1]/sampled_2pt[:,1:])
    Energy=np.mean(Energy_bins,axis=0)


    #jackknife_error=np.sqrt((Nbins-1)/(Nbins))*np.sqrt(np.sum((Energy_bins-Energy)**2,axis=0))
    jackknife_error=np.sqrt((Nbins-1))*np.std(Energy_bins,axis=0)

    return (Energy, jackknife_error, Energy_bins)


def jackknife_1D(l):
    Nbin=len(l)
    jackknife_bins=(np.sum(l)-l)/(Nbin-1)
    jackknife_mean=np.mean(jackknife_bins)
    jackknife_error=np.sqrt(Nbin-1)*np.std(jackknife_bins)
    
    return (jackknife_mean, jackknife_error)


def read_files(base_dir,momentum):
    #Time averaging over the data 
    def time_avg(l):
        n=int(len(l)/2)
        l[1:n]=(l[1:n]+l[-1:-n:-1])/2
        return l[:n+1]
    
    #px^2 + py^2
    def resultant_momentum(px,py):
        return (px**2+py**2)
    
    #reading data of different momentum configuration
    def load_file(file,momentum):
        data=np.loadtxt(file)
        t_final=int(len(data)/64)

        count=0
        d=[]
        for t in range(t_final):
            
            if(resultant_momentum(data[(t*64),0],data[(t*64),1])==momentum):
                d.append(data[t*64:(t+1)*64,4])
                count+=1
                
            elif (resultant_momentum(data[(t*64),0],data[(t*64),1])>momentum):
                break

        result=np.mean(d,axis=0)    
        return result
 
    # Function to average the two files together
    def avg_files(file1, file2,momentum):

        # Read the data from the files
        data1 = load_file(file1,momentum)
        data2 = load_file(file2,momentum)

        result= data1+data2
        return result

    # List to store results from each pair of files
    results = []

    # Loop through the subdirectories
    for folder in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder)
        
        # Ensure that the path is a directory
        if os.path.isdir(folder_path):
            # Find all .dat files in the directory
            dat_files = sorted(glob.glob(os.path.join(folder_path, '*.dat')))
            
            # Check if there are exactly 2 .dat files
            if len(dat_files) == 2:
                file1, file2 = dat_files
                data=avg_files(file1, file2,momentum)
                result = time_avg(data/2)
                results.append(result)
                
            elif len(dat_files) == 1:
                #print(f"Folder {folder} does not contain exactly 2 .dat files.")
                
                data=load_file(dat_files[0],momentum)
                result=time_avg(data)
                results.append(result)
            else:
                pass
            
    return np.array(results)


def platue_function(E,E_error,low, high):
    error_sqaured=1/(E_error[low:high]**2)

    denominator=np.sum(error_sqaured)
    neumerator=np.sum(E[low:high]*error_sqaured)
    return (neumerator/denominator)

def platue_fit(E_bins, Energy_error,low,high):
    platues =[]

    for i in range(len(E_bins)):
        platue_value=platue_function(E_bins[i,:],Energy_error, low,high)
        platues.append(platue_value)
    #print("platue ",np.mean(platues))
    return (jackknife_1D(platues))

