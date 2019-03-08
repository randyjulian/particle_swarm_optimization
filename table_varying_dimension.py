import numpy as np
import math
from statistics import mean,stdev
from sklearn.externals import joblib
import pandas as pd
import os 

# Particle Swarm Optimization Inertia Weight (Fixed Iteration)
def PSO_10k(problem, MaxIter = 10000, PopSize = 32, c1 = 1.4962, c2 = 1.4962, w = 0.9,error=0.01, rosen=False):

    # Empty Particle Template
    empty_particle = {
        'position': None,
        'velocity': None,
        'cost': None,
        'best_position': None,
        'best_cost': None,
    };

    # Extract Problem Info
    CostFunction = problem['CostFunction'];
    VarMin = problem['VarMin'];
    VarMax = problem['VarMax'];
    nVar = problem['nVar'];
    VMax = problem['VMax'];

    # Initialize Global Best
    gbest = {'position': None, 'cost': np.inf};

    # Create Initial Population
    pop=[]
    watch=[]
    for i in range(0, PopSize):
        pop.append(empty_particle.copy());
        pop[i]['position'] = np.random.uniform(VarMin, VarMax, nVar);
        pop[i]['velocity'] = np.zeros(nVar);
        pop[i]['cost'] = CostFunction(pop[i]['position']);
        pop[i]['best_position'] = pop[i]['position'].copy();
        pop[i]['best_cost'] = pop[i]['cost'];
        
        if pop[i]['best_cost'] < gbest['cost']:
            gbest['position'] = pop[i]['best_position'].copy();
            gbest['cost'] = pop[i]['best_cost'];
 
    # PSO Loop
    for it in range(0, MaxIter):
        ## Applying Linearly Decreasing Inertia Weight
        w= (0.9-0.4)*((MaxIter-it)/MaxIter) + 0.4;
        for i in range(0, PopSize):
            
            pop[i]['velocity'] = w*pop[i]['velocity'] \
                + c1*np.random.rand(nVar)*(pop[i]['best_position'] - pop[i]['position']) \
                + c2*np.random.rand(nVar)*(gbest['position'] - pop[i]['position']);
            
            pop[i]['position'] += pop[i]['velocity'];
            ## Setting VMax
            pop[i]['velocity'] = np.maximum(pop[i]['velocity'], -VMax);
            pop[i]['velocity'] = np.minimum(pop[i]['velocity'], VMax);
            
            pop[i]['cost'] = CostFunction(pop[i]['position']);
            
            if pop[i]['cost'] < pop[i]['best_cost']:
                pop[i]['best_position'] = pop[i]['position'].copy();
                pop[i]['best_cost'] = pop[i]['cost'];

        for i in range(0, PopSize):
            if pop[i]['best_cost'] < gbest['cost']:
                gbest['position'] = pop[i]['best_position'].copy();
                gbest['cost'] = pop[i]['best_cost']; 
    
    ## calculating delta_avg and delta_max
    velocity_temp=[]
    position_temp=[]
    delta_avg=0
    delta_max=0

    for i in range(0, PopSize):
        velocity_temp.append(pop[i]['velocity'].copy())
        position_temp.append(pop[i]['position'].copy())
        if rosen == True:
            delta_max=max(delta_max,np.linalg.norm(position_temp[i]-np.ones(nVar)))
        else:
            delta_max=max(delta_max,np.linalg.norm(position_temp[i]))
        delta_avg=delta_avg+np.linalg.norm(velocity_temp[i])

    delta_avg=delta_avg/PopSize
    
    return gbest['cost'],it, delta_avg, delta_max
    
    
# Particle Swarm Optimization Inertia Weight (Stopping Criterion)
def PSO_stop(problem, MaxIter = 100, PopSize = 100, c1 = 1.4962, c2 = 1.4962, w = 0.9, error=0.01, rosen=False):

    # Empty Particle Template
    empty_particle = {
        'position': None,
        'velocity': None,
        'cost': None,
        'best_position': None,
        'best_cost': None,
    };

    # Extract Problem Info
    CostFunction = problem['CostFunction'];
    VarMin = problem['VarMin'];
    VarMax = problem['VarMax'];
    nVar = problem['nVar'];
    VMax = problem['VMax'];

    # Initialize Global Best
    gbest = {'position': None, 'cost': np.inf};

    # Create Initial Population
    pop = [];
    watch=[]
    for i in range(0, PopSize):
        pop.append(empty_particle.copy());
        pop[i]['position'] = np.random.uniform(VarMin, VarMax, nVar);
        pop[i]['velocity'] = np.zeros(nVar);
        pop[i]['cost'] = CostFunction(pop[i]['position']);
        pop[i]['best_position'] = pop[i]['position'].copy();
        pop[i]['best_cost'] = pop[i]['cost'];
        
        if pop[i]['best_cost'] < gbest['cost']:
            gbest['position'] = pop[i]['best_position'].copy();
            gbest['cost'] = pop[i]['best_cost'];
 
    # PSO Loop
    for it in range(0, MaxIter):
        ## applying Linearly Decreasing Inertia Weight
        w= (0.9-0.4)*((MaxIter-it)/MaxIter) + 0.4;
        for i in range(0, PopSize):
            
            ## updating velocity
            pop[i]['velocity'] = w*pop[i]['velocity'] \
                + c1*np.random.rand(nVar)*(pop[i]['best_position'] - pop[i]['position']) \
                + c2*np.random.rand(nVar)*(gbest['position'] - pop[i]['position']);
            
            pop[i]['position'] += pop[i]['velocity'];
            pop[i]['velocity'] = np.maximum(pop[i]['velocity'], -VMax);
            pop[i]['velocity'] = np.minimum(pop[i]['velocity'], VMax);
            
            pop[i]['cost'] = CostFunction(pop[i]['position']);
            
            ## updating pbest
            if pop[i]['cost'] < pop[i]['best_cost']:
                pop[i]['best_position'] = pop[i]['position'].copy();
                pop[i]['best_cost'] = pop[i]['cost'];
        
        ## updating gbest, delta_avg and delta_max
        for i in range(0, PopSize):
            if pop[i]['best_cost'] < gbest['cost']:
                ### If stopping criteria is met
                if (gbest['cost']-pop[i]['best_cost'])<error:
                    velocity_temp=[]
                    position_temp=[]
                    delta_avg=0
                    delta_max=0

                    for i in range(0, PopSize):
                        velocity_temp.append(pop[i]['velocity'].copy())
                        position_temp.append(pop[i]['position'].copy())
                        if rosen == True:
                            delta_max=max(delta_max,np.linalg.norm(position_temp[i]-np.ones(nVar)))
                        else:
                            delta_max=max(delta_max,np.linalg.norm(position_temp[i]))
                        delta_avg=delta_avg+np.linalg.norm(velocity_temp[i])

                    delta_avg=delta_avg/PopSize
                    return gbest['cost'],it,delta_avg,delta_max

                    break   
                gbest['position'] = pop[i]['best_position'].copy();
                gbest['cost'] = pop[i]['best_cost'];

        ## Error catching
        if it==(MaxIter-1):
            print ('Final Iteration exceed max iter, error = {}'.format(gbest['cost']))
            return gbest['cost'],it
            break

### Functions

def Sphere(x):
    return sum(x**2);

def Rastrigrin(x):
    return sum(x**2 - 10*np.cos(2*math.pi*x)+10)

def Rosenbrock(x):
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0) 
    #np.sum(100*(x.T[1:]-x.T[:-1]**2)**2 + (x.T[:-1]-1)**2, axis=0)
    
def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def Griewank(xs):
    sum = 0
    for x in xs:
        sum += x * x
        product = 1
        for i in range(len(xs)):
            product *= math.cos(xs[i] / math.sqrt(i + 1))
    return 1 + sum / 4000 - product
    
def Schaffer(x):
    x_ = x[0]
    y_ = x[1]
    j = 0.5 + (
        (np.sin(x_ ** 2.0 + y_ ** 2.0) ** 2.0 - 0.5)
        / ((1 + 0.001 * (x_ ** 2.0 + y_ ** 2.0)) ** 2.0)
    )
    return j

### Wrapper functions for results
## run_10k and run_stop for running the functions for 50 times
## run_dim_10k and run_dim_stop for running them across different dimension

def run_10k(function,nVar,VarMin,VarMax,VMax,rosen=False):
    problem = {
        'CostFunction': function,
        'nVar': nVar,
        'VarMin': VarMin,   # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
        'VarMax': VarMax,    # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
        'VMax': VMax
    };

    df=[]
    df_it=[]
    df_delta_avg=[]
    df_delta_max=[]
    
    for i in range(0,50):
        print(i)
        cost,it,delta_avg,delta_max=PSO_10k(problem, MaxIter = 10000, PopSize = 32, c1 = 2, c2 = 2, w = 0.9, rosen=rosen)
        df.append(cost)
        df_it.append(it)
        df_delta_avg.append(delta_avg)
        df_delta_max.append(delta_max)
    return df,df_it,df_delta_avg, df_delta_max

def run_stop(function,nVar,VarMin,VarMax,VMax,error, rosen=False):
    problem = {
        'CostFunction': function,
        'nVar': nVar,
        'VarMin': VarMin,   # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
        'VarMax': VarMax,    # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
        'VMax': VMax
    };

    df=[]
    df_it=[]
    df_delta_avg=[]
    df_delta_max=[]
    
    for i in range(0,50):
        print(i)
        cost,it,delta_avg,delta_max=PSO_stop(problem, MaxIter = 10000, PopSize = 32, c1 = 2, c2 = 2, w = 0.9,error=error,rosen=rosen)
        df.append(cost)
        df_it.append(it)
        df_delta_avg.append(delta_avg)
        df_delta_max.append(delta_max)
    return df,df_it ,df_delta_avg, df_delta_max  

def run_dim_10k(function,VarMin,VarMax,VMax,rosen=False):
    cost5,it5,delta_avg_5,delta_max_5=run_10k(function,5,VarMin,VarMax,VMax,rosen=rosen)
    cost10,it10,delta_avg_10,delta_max_10=run_10k(function,10,VarMin,VarMax,VMax,rosen=rosen)
    cost15,it15,delta_avg_15,delta_max_15=run_10k(function,15,VarMin,VarMax,VMax,rosen=rosen)
    cost20,it20,delta_avg_20,delta_max_20=run_10k(function,20,VarMin,VarMax,VMax,rosen=rosen)
    cost25,it25,delta_avg_25,delta_max_25=run_10k(function,25,VarMin,VarMax,VMax,rosen=rosen)
    cost30,it30,delta_avg_30,delta_max_30=run_10k(function,30,VarMin,VarMax,VMax,rosen=rosen)
    
    dictionary_cost={'cost5':cost5,'cost10':cost10,'cost15':cost15,'cost20':cost20,'cost25':cost25,'cost30':cost30}
    dictionary_it={'it5':it5,'it10':it10,'it15':it15,'it20':it20,'it25':it25,'it30':it30}
    dictionary_delta_avg={'delta_avg_5':delta_avg_5,'delta_avg_10':delta_avg_10,'delta_avg_15':delta_avg_15,
                          'delta_avg_20':delta_avg_20,'delta_avg_25':delta_avg_25,'delta_avg_30':delta_avg_30}
    dictionary_delta_max={'delta_max_5':delta_max_5,'delta_max_10':delta_max_10,'delta_max_15':delta_max_15,
                          'delta_max_20':delta_max_20,'delta_max_25':delta_max_25,'delta_max_30':delta_max_30}
    return dictionary_cost,dictionary_it, dictionary_delta_avg,dictionary_delta_max

def run_dim_stop(function,VarMin,VarMax,VMax,error,rosen=False):
    cost5,it5,delta_avg_5,delta_max_5=run_stop(function,5,VarMin,VarMax,VMax,error=error,rosen=rosen)
    cost10,it10,delta_avg_10,delta_max_10=run_stop(function,10,VarMin,VarMax,VMax,error=error,rosen=rosen)
    cost15,it15,delta_avg_15,delta_max_15=run_stop(function,15,VarMin,VarMax,VMax,error=error,rosen=rosen)
    cost20,it20,delta_avg_20,delta_max_20=run_stop(function,20,VarMin,VarMax,VMax,error=error,rosen=rosen)
    cost25,it25,delta_avg_25,delta_max_25=run_stop(function,25,VarMin,VarMax,VMax,error=error,rosen=rosen)
    cost30,it30,delta_avg_30,delta_max_30=run_stop(function,30,VarMin,VarMax,VMax,error=error,rosen=rosen)
    
    dictionary_cost={'cost5':cost5,'cost10':cost10,'cost15':cost15,'cost20':cost20,'cost25':cost25,'cost30':cost30}
    dictionary_it={'it5':it5,'it10':it10,'it15':it15,'it20':it20,'it25':it25,'it30':it30}
    dictionary_delta_avg={'delta_avg_5':delta_avg_5,'delta_avg_10':delta_avg_10,'delta_avg_15':delta_avg_15,
                          'delta_avg_20':delta_avg_20,'delta_avg_25':delta_avg_25,'delta_avg_30':delta_avg_30}
    dictionary_delta_max={'delta_max_5':delta_max_5,'delta_max_10':delta_max_10,'delta_max_15':delta_max_15,
                          'delta_max_20':delta_max_20,'delta_max_25':delta_max_25,'delta_max_30':delta_max_30}
    return dictionary_cost,dictionary_it, dictionary_delta_avg,dictionary_delta_max

### Running PSO and saving it in an array
np.random.seed(21376)
sphere_10k_cost,sphere_10k_it,sphere_10k_d1,sphere_10k_d2=run_dim_10k(Sphere,-100,100,100)
rosenbrock_10k_cost,rosenbrock_10k_it,rosenbrock_10k_d1,rosenbrock_10k_d2=run_dim_10k(Rosenbrock,-30,30,30,True)
rastrigrin_10k_cost,rastrigrin_10k_it,rastrigrin_10k_d1,rastrigrin_10k_d2=run_dim_10k(Rastrigrin,-5.12,5.12,5.12)
griewank_10k_cost,griewank_10k_it,griewank_10k_d1,griewank_10k_d2=run_dim_10k(Griewank,-600,600,600)

sphere_stop_cost,sphere_stop_it,sphere_stop_d1,sphere_stop_d2=run_dim_stop(Sphere,-100,100,100,0.01)
rosenbrock_stop_cost,rosenbrock_stop_it,rosenbrock_stop_d1,rosenbrock_stop_d2=run_dim_stop(Rosenbrock,-30,30,30,100, True)
rastrigrin_stop_cost,rastrigrin_stop_it,rastrigrin_stop_d1,rastrigrin_stop_d2=run_dim_stop(Rastrigrin,-5.12,5.12,5.12,100)
griewank_stop_cost,griewank_stop_it,griewank_stop_d1,griewank_stop_d2=run_dim_stop(Griewank,-600,600,600,0.05)


## creating directory
dirName = 'table dimension files'
os.makedirs(dirName)

## function to dump the results as pickle
def dump_pickle(v1,v2,v3,v4,fn_name,is_10k):
    if is_10k == True:
        joblib.dump(v1,'table dimension files/{}_10k_cost.pkl'.format(fn_name))
        joblib.dump(v2,'table dimension files/{}_10k_it.pkl'.format(fn_name))
        joblib.dump(v3,'table dimension files/{}_10k_d_avg.pkl'.format(fn_name))
        joblib.dump(v4,'table dimension files/{}_10k_d_max.pkl'.format(fn_name))
    else:
        joblib.dump(v1,'table dimension files/{}_stop_cost.pkl'.format(fn_name))
        joblib.dump(v2,'table dimension files/{}_stop_it.pkl'.format(fn_name))
        joblib.dump(v3,'table dimension files/{}_stop_d_avg.pkl'.format(fn_name))
        joblib.dump(v4,'table dimension files/{}_stop_d_max.pkl'.format(fn_name))

dump_pickle(sphere_10k_cost,sphere_10k_it,sphere_10k_d1,sphere_10k_d2,"sphere",True)
#dump_pickle(rosenbrock_10k_cost,rosenbrock_10k_it,rosenbrock_10k_d1,rosenbrock_10k_d2,"rosenbrock",True)
#dump_pickle(rastrigrin_10k_cost,rastrigrin_10k_it,rastrigrin_10k_d1,rastrigrin_10k_d2,"rastrigin",True)
#dump_pickle(griewank_10k_cost,griewank_10k_it,griewank_10k_d1,griewank_10k_d2,"griewank",True)

dump_pickle(sphere_stop_cost,sphere_stop_it,sphere_stop_d1,sphere_stop_d2,"sphere",False)
#dump_pickle(rosenbrock_stop_cost,rosenbrock_stop_it,rosenbrock_stop_d1,rosenbrock_stop_d2,"rosenbrock",False)
#dump_pickle(rastrigrin_stop_cost,rastrigrin_stop_it,rastrigrin_stop_d1,rastrigrin_stop_d2,"rastrigin",False)
#dump_pickle(griewank_stop_cost,griewank_stop_it,griewank_stop_d1,griewank_stop_d2,"griewank",False)


