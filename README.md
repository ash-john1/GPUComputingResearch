# GPU Computing - Undergraduate Engineering Student Research                                     
-	Migrated CPU functions in the GOMC’s open source repository to the GPU 
-	Utilized CUDA to parallelize code operations and improved runtime performance
-	Analyzed the detailed performance metrics of the NVIDIA NSight Compute kernel profiler
-	Contributed code to the GitHub repository in the development branch
-	Utilized Linux, Putty, WinSCP to connect and run commands remotely on university’s GPUs

## Link to repo
https://github.com/GOMC-WSU/GOMC/tree/MoleculeExchange 




## Introduction
The CPU is great for solving core processing tasks in a computer and solving through complex processing tasks with high level of precision. The CPU works by switching from one task to the next. The downside to the CPU is that it can not handle a high workload. For example, tasks such as calculating visual graphics for a video game requires processing many mathematical problems in a short amount of time. The GPU on the other hand is able to compute through these mathematical problems quickly by running the tasks in parallel. 

For my research project, I analyzed how GPUs can be used to improve the performance of molecular computations. I contributed the GOMC project on GitHub. The GOMC project is an open-source software. It simulates molecular systems using the Metropolis Monte Carlo algorithm. Currently, GOMC has code running both on the CPU and the GPU. The goal of my research project was to continue to improve the run time of the computer simulation by utilizing more GPU computation. I worked with Dr. Loren Schwiebert on this project. 

## Project Hypothesis
It is hypothesized that, GPUs can be used to perform numerous similar mathematical operations faster than CPUs by using parallel computing. 

## Research Methods
To conduct a quantitative experiment by writing GPU code and comparing the run time of the GPU to the run time of the CPU. 

## Technologies Used
Throughout my research I learned about how to write GPU code. The programming language I used was C++ . For writing the GPU code, I used CUDA which is a parallel computing platform created by NVIDIA. I learned how to parallelize CPU code by CUDA. I ran the code on Wayne State’s GPUs through the Wayne State grid and connected to it using Putty app. I transferred my files on my local computer to the Grid by using the WinSCP app. To evaluate my code’s performance, I used NVIDIA Nsight Compute which is a kernel profiler for CUDA applications. It provides detailed performance metrics via a user interface and command line tool. After I checked that my code worked, I contributed my code on GitHub to the GOMC repository. I created a MEMC branch off the development branch. 

## MolExchangeReciprocal Function 
I worked on the MolExchangeReciprocal function. This function calculates energy after molecules have been exchanged from one box to another or within the same box. It computes the dot product of the old and new atoms. Then it outputs the difference between the old and new reciprocal energies.  At the start of my research, the MolExchangeReciprocal function only ran on the CPU. It did call a CallMolExchangeReciprocalGPU function but this was a placeholder function. 

During my research, I modified the MolExchangeReciprocal function. In this function I process the charged particle data and pass it to the CallMolExchangeReciprocalGPU function. The main purpose of the CallMolExchangeReciprocalGPU function is to transfer data between the CPU and GPU. In this function, I loaded the variables from the CPU into the GPU and called my GPU function. The name of my GPU function was MolExchangeReciprocalGPUOptimized. This function calculates the dot product value of the atoms and saves the value in a variable. The CallMolExchangeReciprocalGPU function then copies this variable back to the CPU. I used various techniques to optimize the code which I outline in the following sections. 

## Optimizing Calculations 
The main calculation the code performs is dot product. The dot product takes two vectors and multiples each component of first with the corresponding component of the second vector. Then it sums up these values. This calculation offers a way to optimize the code. For example, if one component is zero the final sum will be zero. Therefore, the calculation can be avoided all together if any component is zero.  

In the data processing step of the MolExchangeReciprocal function, I added all the energy of the atoms to the array. However, some of the energy of the atoms are 0. To optimize the code, I decided to only add the charged particles to the array. The benefit of this approach is that it will require less memory to be transferred from CPU and GPU and reduce calculation time. 
	
## Parallelizing Code 
One technique is parallelizing the code. For example, I created a GPU function that looped through each atom for both the new and the old atoms. 

  ```C++
  for (uint p = 0; p < lengthNew; ++p) {
	
    double dotProductNew = DotProductGPU(gpu_kx[threadID], 
                              gpu_ky[threadID],
                              gpu_kz[threadID], 
                              gpu_newMolX[p],
                              gpu_newMolY[p], 
                              gpu_newMolZ[p]);
    double dotsin, dotcos;
    sincos(dotProductNew, &dotsin, &dotcos);
    sumRealNew += (gpu_chargeBoxNew[p] * dotcos);
    sumImaginaryNew += (gpu_chargeBoxNew[p] * dotsin);
  }

  for (uint p = 0; p < lengthOld; ++p) {  
    double dotProductOld = DotProductGPU( gpu_kx[threadID], 
                              gpu_ky[threadID],
                              gpu_kz[threadID], 
                              gpu_oldMolX[p],
                              gpu_oldMolY[p],
                              gpu_oldMolZ[p]);
    double dotsin, dotcos;
    sincos(dotProductOld, &dotsin, &dotcos);
    sumRealNew -= (gpu_chargeBoxOld[p] * dotcos);
    sumImaginaryNew -= (gpu_chargeBoxOld[p] * dotsin);
  }

```

This was similar to the CPU code. To optimize this function, I decided to parallelize the computation on separate threads rather than use a loop. Each thread handled one iteration of the loop. I used the thread ID to determine the atom ID (which is denoted as p) and the image ID. 

   ```C++
  int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadID >= imageSize * chargeBoxLength)
    return;

  // for each new & old atom index, loop thru each image
  int p = threadID / imageSize;
  int imageID = threadID % imageSize;

  double dotProduct = DotProductGPU(…..); 
                                    
  double dotsin, dotcos;
  sincos(dotProduct, &dotsin, &dotcos);

  double sumRealNew = shared_chargeBox[p] * dotcos;
  double sumImaginaryNew = shared_chargeBox[p] * dotsin;

  atomicAdd(&gpu_sumRnew[imageID], sumRealNew);
  atomicAdd(&gpu_sumInew[imageID], sumImaginaryNew);
  ```

The order of which I loop through the atoms also affected the performance. There were two ways to loop through the atoms. One was for each new and old atom, I looped through each image. The other way was for each image, loop through each new and old atom. The former was more efficient because less threads are accessing the same memory location at once. Therefore, there is a short wait time for threads to gain access to the memory.  

## Utilizing Shared Memory
I also worked on optimizing the memory. CUDA provides shared memory. Shared memory is  much faster than local and global memory. It is allocated per thread block, so all threads in the block have access to the same shared memory. Previously in my GPU code, I used local memory. I passed the charged particle arrays as a parameter to the MolExchangeReciprocalGPUOptimized function. 
To make the code more efficient I decided to use dynamic shared memory. I used dynamic rather than static because amount of shared memory is not known at compile time. CUDA has three parameters for the GPU kernel call: blocks per grid, threads per block, and shared memory size. Here is my kernel function call:

  ```C++
 int blocksPerGrid = (int)(totalAtoms / threadsPerBlock) + 1;
  int dynamicSharedMemorySize = 4 * sizeof(double) * (lengthNew + lengthOld);
  MolExchangeReciprocalGPUOptimized <<< blocksPerGrid, threadsPerBlock, dynamicSharedMemorySize>>>( …
  ```

In my MolExchangeReciprocalGPUOptimized function, I create the dynamic shared memory which is named as shared_arr. Then I declared two arrays shared_chargeBox and shared_Mol.

  ```C++
  extern __shared__ double shared_arr[]; 
  double* shared_chargeBox = (double*)shared_arr; 
  double* shared_Mol = (double*)&shared_chargeBox[chargeBoxLength]; 

Each thread is responsible for loading one index of the array as shown in the code below. 
  if(threadIdx.x < lengthNew) { 
    shared_Mol[threadIdx.x * 3] = gpu_newMolX[threadIdx.x];
    shared_Mol[threadIdx.x * 3 + 1] = gpu_newMolY[threadIdx.x];
    shared_Mol[threadIdx.x * 3 + 2] = gpu_newMolZ[threadIdx.x];
    shared_chargeBox[threadIdx.x] = gpu_chargeBoxNew[threadIdx.x];
  }
  else if (threadIdx.x < chargeBoxLength) {
    int gpu_oldMolIndex = threadIdx.x - lengthNew;
    shared_Mol[threadIdx.x * 3] = gpu_oldMolX[gpu_oldMolIndex];
    shared_Mol[threadIdx.x * 3 + 1] = gpu_oldMolY[gpu_oldMolIndex];
    shared_Mol[threadIdx.x * 3 + 2] = gpu_oldMolZ[gpu_oldMolIndex];
    shared_chargeBox[threadIdx.x] = -gpu_chargeBoxOld[gpu_oldMolIndex];
  }
  __syncthreads();
  ```

The __syncthreads function is called at the end which waits for every thread to finish loading the array before proceeding to the next line of code. Using shared memory improved the run time of the code. 

## Evaluating Results
To check if my GPU code is outputting the correct results, I ran the code using the configuration file. This configuration file sets up the initial values of different variables. I ran the configuration file in both the CPU and the GPU. The results were outputted to a text file. I compared the energy values of both results. If my GPU code ran correctly, it should match the results of the CPU. If it did not match, I edit the code and run it again. To check the performance of my GPU code, I ran the code through the NVIDIA profiler. The profiler outputs a file which contains runtime and other performance metrics. Overall, the CPU code ran an average of 7.1342 seconds and the GPU ran an average of 4.8590 seconds. 

## Conclusion 
By running the code on the GPU and using optimization techniques, I improved the run time and the memory utilization of the MolExchangeReciprocal function. 

## Acknowledgements
I thank the Wayne State College of Engineering for giving me this opportunity to research and expand my knowledge. I also thank Dr. Loren Schwiebert who has helped me through this project.

## References
CUDA toolkit documentation v11.6.2. CUDA Toolkit Documentation. (n.d.). Retrieved April 18, 2022, from https://docs.nvidia.com/cuda/
Y. Nejahi, M. Soroush Barhaghi, G. Schwing, L. Schwiebert, J. Potoff. SoftwareX, 13, 100627 (2021)
Y. Nejahi, M. Soroush Barhaghi, J. Mick, B. Jackman, K. Rushaidat, Y. Li, L. Schwiebert, J. Potoff. SoftwareX, 9, 20-27 (2019)

