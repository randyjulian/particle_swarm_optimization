# Effects of Parameter Choices on Convergence Behaviour in Particle Swarm Optimization
Final Year Project for Bachelor of Science (Honours), Statistics

## Abstract
Particle Swarm Optimization (PSO) is a metaheuristic, nature-inspired algorithm commonly used for solving optimization problems. One advantage of PSO is the fast convergence of the algorithm but is also known for its tendency for premature convergence and stagnation. This study builds on the empirical study by Cleghorn and Engelbrecht (2014) that studies the region of convergence which is measured by the precision of the algorithm. In Chapter 3, this study will explore a new measure to assess accuracy to complement the existing precision measure. Combining these two measures, a new derived region for convergence is introduced and further examination of non-convergence behaviour is conducted. There are 3 main categories for the behaviour of the algorithm: convergence, stagnation and oscillation. While the choices of parameters have a huge impact on the stagnation and oscillation cases, it can be shown that in certain cases of inaccurate convergence, longer iterations can help the algorithm to converge.

## Files
The scripts should be run first to obtain the data to be used in the notebooks later (the two table scripts take about 6 hours). The two notebooks: convergence_plots_chapter5 and non_convergence_plots_chapter6 require the pickle files from convergence_matrices_creation.py in order to run.

* Python Scripts
    * table_varying_dimension.py
    * table_varying_particles.py
    * convergence_matrices_creation.py

* Jupyter Notebooks
    * reading_table_files.ipynb
    * convergence_plots_chapter5.ipynb
    * non_convergence_plots_chapter6.ipynb

## Table of Contents
1. **Introduction**  
2. **Particle Swarm Optimization**  
    2.1. Minimization problem  
    2.2. Algorithm  
     * 2.2.1. Pseudo Code  
     * 2.2.2. Initialization  
     * 2.2.3. Position and Velocity Update
     * 2.2.4. Parameters
     * 2.2.5. Restrictions
    2.3 Exploration Exploitation Trade-off
    2.4. Applications of PSO
3. **Experimental Setup**  
    3.1. Objective Functions  
    3.2. Measure of Convergence  
     * 3.2.1. Fixed Iteration and Stopping Criterion Methods
     * 3.2.2. Measure of Convergence, $\Delta_{avg}$
     * 3.2.3. New Measure of Convergence, $\Delta_{max}$
     * 3.2.4. Benefits of $\Delta_{avg}$ and $\Delta_{max}$
4. **Understanding Convergence through Fixed Iteration and Stopping Criteria Methods**  
    4.1. Simulation with Varying Number of Particles  
    4.2. Simulation with Varying Dimensions  
    4.3. Conclusion  
5. **Effects of PSO Parameters on its Convergence Behaviour**  
    5.1. Theoretical Regions of Particle Convergence  
    5.2. Experimental Setup  
    5.3. Region of Convergence based on $\Delta_{avg}$  
     * 5.3.1. Classification of Convergence Behaviour
     * 5.3.2. Classification Method
     * 5.3.3. Pseudo Code
     * 5.3.4. Limitations of Classification Method
     * 5.3.5. Results
    5.4. Region of Convergence based on $\Delta_{max}$  
     * 5.4.1. Results  
     * 5.4.2. Limitations  
    5.5. Conclusion   
6. **Non-Convergence Behaviour**  
    6.1. Approach on Observing Non-Convergence Behaviour  
    6.2. Inaccurate Convergence (ω= 0.4, $c_1+c_2$= 4.4)  
    6.3. Stagnation (ω= 1.0, $c_1+c_2$= 1.2)  
    6.4. Oscillation (ω=0.9,$c_1+c_2$=3.8)  
    6.5. Conclusion  
7. **Conclusion**  
    7.1. Results   
    7.2. Limitations and Future Work  


## Author
Please contact the author at randyjulian.ykl@gmail.com for the paper

## Acknowledgements
* Yarpiz for the basic implementation of PSO (http://yarpiz.com/463/ypea127-python-implementation-particle-swarm-optimization-pso)
