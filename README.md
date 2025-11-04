# Advanced-Computational-Framework-for-Terahertz-Inverse-Imaging-in-Stratified-Biomedical-Media
# Paper Description: 
Novel 2D physics-informed Inverse Image reconstruction of skin permittivity using THz-TDS and FDTD  
**First 2D Inverse THz Imaging of skin hydration Gradient for the ε_r range of 6-9**

# Programs Descriptions: 
**Below described files are to be obligatorily simulated in MATLAB environment. All the files are to be simulated in the exact order of mention below**   
**The output fot all the programs simulated in MATLAB can be extracted in a graphical as well as xlsx format and can be later used for final Simulation in Python Environment.**   

File **"THz_Pulse_Sourse.m"** Defines the mathematical aspect for the required input Pulse. The generates Broadband THz pulse used in FDTD simulation later (Left output Graph). It also defines a Time-domain differentiated Gaussian pulse centered at 1 THz, which will be later used in the simulation. It also defines a Frequency spectrum showing 0.1–2.5 THz bandwidth, consistent with typical THz-TDS systems(Right output Graph). The pulse's behavior and path can be witnessed and traced via the output graph generated on running the file.  

File **"skin_Hydration_ground_truth.m"** defines Biophysical hydration profile in human skin. The generated graph shows the dependency of water volume fraction to the depth of the skin. 

File **"Dielectric_permitivity_vs_Frequency.m"** depicts the **Dual-Debye model of skin permittivity.** The generated output graph depicts the dependency of the dielectric permitivity to the frequency used. 


**Below described files are to be simulated in a Python environment. All the files are to be simulated in the exact order of mention below** 
File **"FDTD_Simulation_Reflected_pulse_Echo_timing_TV_reconstructions.py"** contains the python code to simulate FDTD on the created model (Figure 1 of the output) along with calculating echo timing and generating a realistic reflected pulse (Figure 2 of the output) for the system. All the data calculated and generated is at the end used to for Total Variation (TV) Reconstruction (Figure 3 of the output)
