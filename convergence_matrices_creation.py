import numpy as np
import math
from sklearn.externals import joblib
import os

# Particle Swarm Optimization Inertia Weight
def PSO(problem, MaxIter = 2000, PopSize = 32,w=0,c1=0,c2=0,rosen=False):

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
    init_pos=[]
    total=[]
    np.random.seed(23451)
    for i in range(0, PopSize):
        pop.append(empty_particle.copy());
        pop[i]['position'] = np.random.uniform(VarMin, VarMax, nVar);
        pop[i]['velocity'] = np.zeros(nVar);
        pop[i]['cost'] = CostFunction(pop[i]['position']);
        pop[i]['best_position'] = pop[i]['position'].copy();
        pop[i]['best_cost'] = pop[i]['cost'];
        init_pos.append(pop[i]['position'].copy())
        
        if pop[i]['best_cost'] < gbest['cost']:
            gbest['position'] = pop[i]['best_position'].copy();
            gbest['cost'] = pop[i]['best_cost'];
 
    # PSO Loop
    delta_avg_list=[]
    delta_max_list=[]
    delta_max_50=[]
    pop_final=[]
    for it in range(0, MaxIter):
       
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


        ## saving observations every 40 iterations to save space
        if it % 40 == 0:
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

    
            delta_avg_list.append(delta_avg)
            delta_max_list.append(delta_max) 
        
        ## saving last 50 iteration for delta_max
        if it > 1949:
            
            delta_max=0
            position_temp=[]
            for i in range(0, PopSize):
                position_temp.append(pop[i]['position'].copy())
                if rosen == True:
                    delta_max=max(delta_max,np.linalg.norm(position_temp[i]-np.ones(nVar)))
                else:
                    delta_max=max(delta_max,np.linalg.norm(position_temp[i]))
            delta_max_50.append(delta_max)
    
    ## saving final position of the pasticles
    if it == 1999:
        for i in range(0,PopSize):
            pop_final.append(pop[i]['position'].copy())
                
    return [delta_avg_list,delta_max_list,delta_max_50,pop_final]

## defining functions

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
        product *= math.cos(float(xs[i]) / math.sqrt(i + 1))
    return 1 + sum / 4000 - product
    
def Schaffer(x):
    x_ = x[0]
    y_ = x[1]
    j = 0.5 + ((np.sin(math.sqrt(x_ ** 2.0 + y_ ** 2.0))) ** 2.0 - 0.5)/((1 + 0.001 * (x_ ** 2.0 + y_ ** 2.0)) ** 2.0)
    return j

## setting PSO prblem definition
sphere_problem = {
        'CostFunction': Sphere,
        'nVar': 10,
        'VarMin': -100,   # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
        'VarMax': 100,    # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
        'VMax': 100
    };

rosenbrock_problem = {
        'CostFunction': Rosenbrock,
        'nVar': 10,
        'VarMin': -100,   # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
        'VarMax': 100,    # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
        'VMax': 100
    };

schaffer_problem = {
        'CostFunction': Schaffer,
        'nVar': 2,
        'VarMin': -100,   # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
        'VarMax': 100,    # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
        'VMax': 100
    };

rastrigin_problem = {
        'CostFunction': Rastrigrin,
        'nVar': 10,
        'VarMin': -5.12,   # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
        'VarMax': 5.12,    # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
        'VMax': 5.12
    };

griewank_problem = {
        'CostFunction': Griewank,
        'nVar': 10,
        'VarMin': -600,   # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
        'VarMax': 600,    # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
        'VMax': 600
    };

## saving the 12x45 matrix
def matrix1245(problem,rosen):
    avg_res=np.zeros([12,45],dtype=np.ndarray)
    max_res=np.zeros([12,45],dtype=np.ndarray)
    delta_max_50=np.zeros([12,45],dtype=np.ndarray)
    final_pos=np.zeros([12,45],dtype=np.ndarray)

    temp=[]
    w=0
    c=0
    for i in range(0,12):
        _w=i/10
        for j in range (0,45):
            _c=j/20
            print(_w,_c)
            print(i,j)
            temp=PSO(problem, MaxIter = 2000, PopSize = 32, w=_w, c1=_c,c2=_c,rosen=rosen)
            avg_res[i][j]=temp[0]
            max_res[i][j]=temp[1]
            delta_max_50[i][j]=temp[2]
            final_pos[i][j]=temp[3]
    return avg_res,max_res,delta_max_50,final_pos

## creating directories
dirName = 'convergence pickle files'
os.makedirs(dirName)

sphere_avg,sphere_max,sphere_50,sphere_pos=matrix1245(sphere_problem,False)
joblib.dump([sphere_avg,sphere_max,sphere_50,sphere_pos],"convergence pickle files/sphere_all.pkl")

rastrigin_avg,rastrigin_max,rastrigin_50, rastrigin_pos=matrix1245(rastrigin_problem,False)
joblib.dump([rastrigin_avg,rastrigin_max,rastrigin_50, rastrigin_pos],"convergence pickle files/rastrigin_all.pkl")

griewank_avg,griewank_max,griewank_50,griewank_pos=matrix1245(griewank_problem,False)
joblib.dump([griewank_avg,griewank_max,griewank_50,griewank_pos],"convergence pickle files/griewank_all.pkl")

schaffer_avg,schaffer_max,schaffer_50,schaffer_pos=matrix1245(schaffer_problem,False)
joblib.dump([schaffer_avg,schaffer_max,schaffer_50,schaffer_pos],"convergence pickle files/schaffer_all.pkl")

rosen_avg,rosen_max,rosen_50,rosen_pos=matrix1245(rosenbrock_problem,True)
joblib.dump([rosen_avg,rosen_max,rosen_50,rosen_pos],"convergence pickle files/rosen_all.pkl")