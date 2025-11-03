# Advanced-Computational-Framework-for-Terahertz-Inverse-Imaging-in-Stratified-Biomedical-Media
# Paper Description: 
Novel 2D physics-informed Inverse Image reconstruction of skin permittivity using THz-TDS and FDTD  
**First 2D Inverse THz Imaging of skin hydration Gradient for the ε_r range of 6-9**

# Programs Descriptions: 
**Below described files are to be obligatorily simulated in MATLAB environment.**
**The output fot all the programs simulated in MATLAB can be extracted in a graphical as well as xlsx format and can be later used for final Simulation in Python Environment.**

File **"THz_Pulse_Sourse.m"** Defines the mathematical aspect for the required input Pulse. The pulse's behavior and path can be witnessed and traced via the output graph generated on running the file.


**Below described files are to be simulated in a Python environment.** 
File **"FDTD_Simulation_Reflected_pulse_Echo_timing_TV_reconstructions.py"** contains the python code to simulate FDTD on the created model (Figure 1 of the output) along with calculating echo timing and generating a realistic reflected pulse (Figure 2 of the output) for the system. All the data calculated and generated is at the end used to for Total Variation (TV) Reconstruction (Figure 3 of the output)
