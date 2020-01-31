'''
This is hosted on github.

This is heavily based on the example from Artificial Intelligence: A Modern Approach located here:
http://aima.cs.berkeley.edu/python/agents.html
http://aima.cs.berkeley.edu/python/agents.py
'''

import inspect
import random, copy, warnings
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from tqdm import tqdm
import concurrent.futures
import pickle

# import my files
from agents import *
from objects import *
from display import *
from problem import *
from comms import *
import json
import utils

'''Implement Agents and Environments (Chapters 1-2).

The class hierarchies are as follows:

Object ## A physical object that can exist in an environment
    Agent
        RandomReflexAgent
        ...
    Dirt
    Wall
        DeadCell
    Fire
    ...

Environment ## An environment holds objects, runs simulations
    XYEnvironment
        VacuumEnvironment

EnvFrame ## A graphical representation of the Environment

'''

# TODO: Fix the issue with the seed not functioning as expected for multi-run tests (e.g. test9())

def new_seed(a=None): # create a new_seed function to add new behavior before old_seed
    if a == None:                   # if the seed is random
        print('seed is None, generating random seed')
        a = random.getrandbits(16)  # then create a random 20 bit number (1 in 1,000,000)
    random.current_seed = a         # store the seed value in random.current_seed if hasattr(random,'current_seed') else '??' if hasattr(random,'current_seed') else '??'
    old_seed(a)                     # run the old_seed function with the seed value (a)

#random.seed = new_seed  # replace the random.seed function with our new_seed function
#random.seed(None)       # generate a random seed to use have a known seed as the current value


current_seed = None
def set_seed(a=None):
    if a == None:  # if the seed is random
        print('seed is None, generating random seed')
        a = random.getrandbits(16)  # then create a random 20 bit number (1 in 1,000,000)
    global current_seed
    current_seed = a  # store the seed value in current_seed if hasattr(random,'current_seed') else '??' if hasattr(random,'current_seed') else '??'
    random.seed(a)


class Environment:
    """
    Abstract class representing an Environment.  'Real' Environment classes inherit from this. Your Environment will
    typically need to implement:
        percept:           Define the percept that an agent sees.
        execute_action:    Define the effects of executing an action.
                           Also update the agent.performance slot.
    The environment keeps a list of .objects and .agents (which is a subset of .objects). Each agent has a .performance
    slot, initialized to 0.  Each object has a .location slot, even though some environments may not need this.
    """

    def __init__(self,):
        self.t = 0
        self.objects = []
        self.agents = []
        self.perceptors = {}
        self.communicator = None
        self.actuators = {}
        self.problem = None

    # Mark: What does this do?  It isn't checked in the Environment class's add_object.
    object_classes = [] ## List of classes that can go into environment

    def percept(self, agent):
        agentpercept = {}  # initialize the percept dictionary
        for per in agent.perceptor_types:  # for each perceptor in agent
            # calculate the percept value for the perceptor and append to the percept dictionary
            agentpercept.update(self.perceptors[per.__name__].percept(agent))
        return agentpercept

    def execute_action(self, agent, action, params=None):
        if not params: params = {}
        if [act.type for act in agent.actuator_types if type(self.actuators[action]) is act.type]:  # if the requested action is in the list
            for ps in [c.params for c in agent.actuator_types if c.type.__name__ == action]:
                params.update(ps)
            self.actuators[action].actuate(agent, params)  # call the requested action from the actuators dictionary
        else:
            # if the agent is trying to do something it can't, raise a warning
            warnings.warn('%s requested action %s and it is not supported.' % (agent, action))

    def communicate(self, from_agent):
        if self.communicator:
            agents_seen = self.communicator.get_comms_network(from_agent)
            for to_agent in agents_seen:
                self.communicator.communicate(from_agent.percepts, from_agent, to_agent)

    def default_location(self, obj):
        "Default location to place a new object with unspecified location"
        return None


    def exogenous_change(self):
	    "If there is spontaneous change in the world, override this."
	    pass


    def is_done(self):
        "By default, we're done when we can't find a live agent."
        for agent in self.agents:
            if agent.is_alive(): return False
        return True


    def step(self):
        '''Run the environment for one time step. If the
        actions and exogenous changes are independent, this method will
        do.  If there are interactions between them, you'll need to
        override this method.'''
        if not self.is_done():
            # increment time counter
            self.t += 1

            for agent in self.agents:
                agent.percepts = self.percept(agent)

            self.communicator.setup()
            for from_agent in self.agents:  # TODO: how to add communication as an action?
                self.communicate(from_agent)

            for agent in self.agents:
                if hasattr(agent, 'state_estimator'):
                    if hasattr(agent, 'state'):
                        agent.state = agent.state_estimator(agent.percepts, agent.comms, state=agent.state)
                    else:
                        agent.percepts = agent.state_estimator(agent.percepts, agent.comms)    # is there anything that we want to do here?
                else:       # if there is no state_estimator() then just passthrough the percepts to the state
                    #agent.state = agent.percepts
                    pass

            # for each agent
            # run agent.program with the agent's state as an input
            # agent's perception = Env.state(agent)

            # generate actions
            if hasattr(agent, 'state'):
                actions = [agent.program(agent.state) for agent in self.agents]
            else:
                actions = [agent.program(agent.percepts) for agent in self.agents]


            # for each agent-action pair, have the environment process the actions
            for (agent, action) in zip(self.agents, actions):
                self.execute_action(agent, action)

            # process any external events
            self.exogenous_change()

    def run(self, steps=1000):
        for step in range(steps): # Run the Environment for given number of time steps.
            if self.is_done(): return
            self.step()

    def add_object(self, obj, location=None):
        '''Add an object to the environment, setting its location. Also keep track of objects that are agents.
        Shouldn't need to override this.'''
        obj.location = location or self.default_location(obj)
        # Mark: ^^ unsure about this line, lazy evaluation means it will only process if location=None?
        # Add the new Object to self.objects
        self.objects.append(obj)
        # If the object is an Agent, add it to self.agents and initialize performance parameter
        if isinstance(obj, Agent):
            obj.performance = 0
            self.add_perceptor_for_agent(obj)
            self.add_actuator_for_agent(obj)
            self.add_communicator_for_agent(obj)
            self.agents.append(obj)
        return obj

    def add_perceptor_for_agent(self, agent):
        for pertype in agent.perceptor_types: # for each type of perceptor for the agent
            if not [p for p in self.perceptors.values() if type(p) is pertype]: # if the perceptor doesn't exist yet
                self.perceptors[pertype.__name__] = pertype(self) # add the name:perceptor pair to the dictionary

    def add_communicator_for_agent(self, agent):
        if hasattr(agent, 'communicator') and agent.communicator:
            if self.communicator: # if the communicator exists...
                if not type(self.communicator) is agent.communicator:
                    # if the communicator exists, but is a different type, throw and error (TODO: implement multiple communicators)
                    raise ValueError('Communicator already exists')
                else:  # if communicator exists and is the same type, don't recreate, just pass
                    pass
            else: # if it doesn't exist, create a new communicator based on the agent's communicator definition
                self.communicator = agent.communicator(self) # set the communicator equal to the agent communicator

    def add_actuator_for_agent(self, agent):
        for acttype in agent.actuator_types: # for each type of perceptor for the agent
            if not [act for act in self.actuators.values() if type(act) is acttype.type]: # if the perceptor doesn't exist yet
                self.actuators[acttype.type.__name__] = acttype.type(self) # add the name:perceptor pair to the dictionary


class XYEnvironment(Environment):
    '''This class is for environments on a 2D plane, with locations
    labelled by (x, y) points, either discrete or continuous.  Agents
    perceive objects within a radius.  Each agent in the environment
    has a .location slot which should be a location such as (0, 1),
    and a .holding slot, which should be a list of objects that are
    held '''

    #robot_images = {(1,0):'img/robot-right.gif',(-1,0):'img/robot-left.gif',(0,-1):'img/robot-up.gif',(0,1):'img/robot-down.gif'}

    def __init__(self, width=10, height=10):
        # set all of the initial conditions with the update function
        self.width = width
        self.height = height
        Environment.__init__(self)

    def objects_of_type(self, cls):
        # Use a list comprehension to return a list of all objects of type cls
        return [obj for obj in self.objects if isinstance(obj, cls)]

    def objects_at(self, location):
        "Return all objects exactly at a given location."
        return [obj for obj in self.objects if obj.location == location]

    def find_at(self, cls, loc):
        return [o for o in self.objects_at(loc) if isinstance(o, cls)]

    def objects_near(self, location, radius):
        """Return all objects within radius of location."""
        radius2 = radius * radius # square radius instead of taking the square root for faster processing
        return [obj for obj in self.objects if isinstance(obj.location, tuple) and distance(location, obj.location) <= radius]

    def default_location(self, obj):
        # If no location is specified, set the location to be a random location in the Environment.
        return (random.choice(self.width), random.choice(self.height))

    def move_to(self, obj, destination):
        """Move an object to a new location."""
        # Currently move_to assumes that the object is only moving a single cell at a time
        # e.g. agent.location + agent.heading => (x,y) + (0,1)
        #
        # The function finds all objects at the destination that have the blocker flag set.
        # If there are none, move to the destination

        obstacles = [o for o in self.objects_at(destination) if o.blocker]
        if not obstacles:
            obj.location = destination


    def add_object(self, obj, location=(1, 1)):
        Environment.add_object(self, obj, location)

        obj.holding = []
        obj.held = None

        return obj

    def add_walls(self):
        "Put walls around the entire perimeter of the grid."
        for x in range(self.width-1):
            self.add_object(Wall(), (x, 0))
            self.add_object(Wall(), (x+1, self.height-1))
        for y in range(self.height-1):
            self.add_object(Wall(), (0, y+1))
            self.add_object(Wall(), (self.width-1, y))


#______________________________________________________________________________
## Vacuum environment

class VacuumEnvironment(XYEnvironment):
    '''The environment of [Ex. 2.12]. Agent perceives dirty or clean,
    and bump (into obstacle) or not; 2D discrete world of unknown size;
    performance measure is 100 for each dirt cleaned, and -1 for
    each turn taken.'''
    def __init__(self, width=10, height=10):
        XYEnvironment.__init__(self, width, height)
        self.add_walls()

    object_classes = []

    def exogenous_change(self):
        pass

def NewVacuumEnvironment(width=10, height=10, config=None):
    e = VacuumEnvironment(width=width, height=height)
    # Generate walls with dead cells in the center
    if config==None:
        pass
    elif config=='empty':
        # no dirt
        # extend exogenous_change with function to detect if no dirt is left
        old_exogenous_chage = e.exogenous_change

        def new_exogenous_change(self):
            old_exogenous_chage()
            if not [d for d in self.objects_of_type(Dirt) if isinstance(d.location, tuple)]:
                for a in self.agents:
                    a.alive = False
                    a.performance = self.t

        e.exogenous_change = MethodType(new_exogenous_change, e)
    elif config == 'shape of eight':
        for x in [2,3]:
            for y in [2,3]:
                e.add_object(Wall(), (x,y))
        for x in [2,3]:
            for y in [5,6]:
                e.add_object(Wall(), (x,y))
        e.add_object(Dirt(),location=(4,5))
    elif config=='center walls':
        for x in range(int(e.width/2-5),int(e.width/2+5)):
            for y in range(int(e.height/2-5),int(e.height/2+5)):
                if ((x == int(e.width/2-5)) or (x == int(e.width/2+4)) or
                    (y == int(e.height/2-5)) or (y == int(e.height/2+4))):
                    e.add_object(Wall(), (x,y))
                else:
                    e.add_object(DeadCell(), (x,y))

    elif config=='full dirt':
        # Fill a square area with dirt
        for x in range(0,e.width):
            for y in range(0,e.height):
                if not e.find_at(Wall,(x,y)): e.add_object(Dirt(),location=(x,y))

        # extend exogenous_change with function to detect if no dirt is left
        old_exogenous_chage = e.exogenous_change
        def new_exogenous_change(self):
            old_exogenous_chage()
            if not [d for d in self.objects_of_type(Dirt) if isinstance(d.location, tuple)]:
                for a in self.agents:
                    a.alive = False
                    a.performance = self.t

        e.exogenous_change = MethodType(new_exogenous_change, e)

    elif config=='sparse dirt':
        # Fill a square area with dirt every n cells
        stp = 3
        for x in range(0,e.width,stp):
            for y in range(0,e.height,stp):
                if not e.find_at(Wall,(x,y)): e.add_object(Dirt(),location=(x,y))

        # extend exogenous_change with function to detect if no dirt is left
        old_exogenous_chage = e.exogenous_change
        def new_exogenous_change(self):
            old_exogenous_chage()
            if not [d for d in self.objects_of_type(Dirt) if isinstance(d.location, tuple)]:
                for a in self.agents:
                    a.alive = False
                    a.performance = self.t

        e.exogenous_change = MethodType(new_exogenous_change, e)

    elif config=='corner dirt':
        # Fill a square area with dirt every n cells
        for (dx, dy) in [(0, 0), (37, 0), (0, 37), (37, 37)]:
            for x in range(1, 12, 2):
                for y in range(1, 12, 2):
                    if not e.find_at(Wall, (dx+x, dy+y)): e.add_object(Dirt(), location=(dx+x, dy+y))

        # extend exogenous_change with function to detect if no dirt is left
        old_exogenous_chage = e.exogenous_change

        def new_exogenous_change(self):
            old_exogenous_chage()
            if not [d for d in self.objects_of_type(Dirt) if isinstance(d.location, tuple)]:
                for a in self.agents:
                    a.alive = False
                    a.performance = self.t

        e.exogenous_change = MethodType(new_exogenous_change, e)

    elif config=='random dirt':
        for x in range(100):
            loc = (random.randrange(width), random.randrange(height))
            if not (e.find_at(Dirt, loc) or e.find_at(Wall, loc)):
                e.add_object(Dirt(), loc)
        # extend exogenous_change with function to detect if no dirt is left
        old_exogenous_chage = e.exogenous_change
        def new_exogenous_change(self):
            old_exogenous_chage()
            if not [d for d in self.objects_of_type(Dirt) if isinstance(d.location, tuple)]:
                for a in self.agents:
                    a.alive = False
                    a.performance = self.t

        e.exogenous_change = MethodType(new_exogenous_change, e)

    elif config=='random dirt and wall':
        for x in range(int(e.width/2-5),int(e.width/2+5)):
            for y in range(int(e.height/2-5),int(e.height/2+5)):
                if ((x == int(e.width/2-5)) or (x == int(e.width/2+4)) or
                    (y == int(e.height/2-5)) or (y == int(e.height/2+4))):
                    e.add_object(Wall(), (x,y))
        for x in range(50):
            loc = (random.randrange(width), random.randrange(width))
            if not (e.find_at(Dirt, loc) or e.find_at(Wall, loc) or (loc[0] > 5 and loc[0]< 14) and loc[1] > 5 and loc[1] < 14):
                e.add_object(Dirt(), loc)

    elif config=='center walls w/ random dirt and fire':
        for x in range(int(e.width/2-5),int(e.width/2+5)):
            for y in range(int(e.height/2-5),int(e.height/2+5)):
                if ((x == int(e.width/2-5)) or (x == int(e.width/2+4)) or
                    (y == int(e.height/2-5)) or (y == int(e.height/2+4))):
                    e.add_object(Wall(), (x,y))
                else:
                    e.add_object(DeadCell(), (x,y))

        # adds custom behavior to the exogenous_chage() method to avoid creating a new class
        # is that correct?  should we just create a new class?

        def exogenous_dirt(self):
            if random.uniform(0, 1) < 1.0:
                loc = (random.randrange(self.width), random.randrange(self.height))
                if not (self.find_at(Dirt, loc) or self.find_at(Wall, loc)):
                    self.add_object(Dirt(), loc)

        def exogenous_fire(self):
            fs = self.objects_of_type(Fire)

            if fs:
                for f in fs:
                    if f.t == 0:
                        f.destroy()
                        self.objects.remove(f)
                    else:
                        f.t -= 1
                        if random.uniform(0, 1) < 0.21:
                            emptyCells = [(x, y) for x in range(f.location[0] - 1, f.location[0] + 2)
                                          for y in range(f.location[1] - 1, f.location[1] + 2)
                                          if not self.objects_at((x, y))]
                            if emptyCells: self.add_object(Fire(), random.choice(emptyCells))
            else:  # if there is no fire
                for i in range(5):
                    for i in range(10):  # try 10 times, would do while, but that could get stuck
                        loc = (random.randrange(1, self.width), random.randrange(1, self.width))
                        if not self.objects_at(loc):
                            self.add_object(Fire(), loc)
                            break

        old_exogenous_chage = e.exogenous_change
        def new_exogenous_change(self):
            old_exogenous_chage()
            exogenous_dirt(self)
            exogenous_fire(self)

        e.exogenous_change = MethodType(new_exogenous_change, e)

    return e
#______________________________________________________________________________

def compare_agents(EnvFactory, AgentFactories, n=10, steps=100):
    '''See how well each of several agents do in n instances of an environment.
    Pass in a factory (constructor) for environments, and several for agents.
    Create n instances of the environment, and run each agent in copies of
    each one for steps. Return a list of (agent, average-score) tuples.'''
    envs = [EnvFactory() for i in range(n)]
    return [(A, test_agent(A, steps, copy.deepcopy(envs)))
            for A in AgentFactories]

def test_agent(AgentFactory, steps, envs):
    "Return the mean score of running an agent in each of the envs, for steps"
    total = 0
    i = 0
    for env in envs:
        i+=1
        with Timer(name='Simulation Timer - Agent=%s' % i, format='%.4f'):
            agent = AgentFactory()
            env.add_object(agent)
            env.run(steps)
            total += env.t
    return float(total)/len(envs)

#______________________________________________________________________________


def test0(seed=None):
    # set a seed to provide repeatable outcomes each run
    set_seed(seed) # if the seed wasn't set in the input, the default value of none will create (and store) a random seed

    e = NewVacuumEnvironment(width=20,height=20,config="random dirt")
    ef = EnvFrame(e,root=tk.Tk(),cellwidth=30,
                    title='Vacuum Robot Simulation - Scenario=%s(), Seed=%s' % (inspect.stack()[0][3],current_seed))

    # Create agents

    e.add_object(GreedyAgentWithRangePerception(sensor_radius=3))

    ef.configure_display()
    # ef.run()
    ef.mainloop()


def test1(seed=None):
    # set a seed to provide repeatable outcomes each run
    set_seed(seed) # if the seed wasn't set in the input, the default value of none will create (and store) a random seed

    e = NewVacuumEnvironment(width=20,height=20,config="center walls w/ random dirt and fire")
    ef = EnvFrame(e,root=tk.Tk(),cellwidth=30,
                    title='Vacuum Robot Simulation - Scenario=%s(), Seed=%s' % (inspect.stack()[0][3],current_seed))

    # Create agents
    for i in range(1,19):
        e.add_object(NewRandomReflexAgent(debug=False),location=(1,i)).id = i

    ef.configure_display()
    ef.run()
    ef.mainloop()


def test2(seed=None):
    # set a seed to provide repeatable outcomes each run
    set_seed(seed) # if the seed wasn't set in the input, the default value of none will create (and store) a random seed

    EnvFactory = partial(NewVacuumEnvironment,width=20,height=20,config="random dirt")
    sensor_radii = range(10)
    results = compare_agents(EnvFactory, [partial(NewGreedyAgentWithRangePerception, debug=False, sensor_radius=r) for r in sensor_radii], n=10, steps=2000)
    print(results)
    plt.plot(sensor_radii,[r[1] for r in results],'r-')
    plt.title('scenario=%s(), seed=%s' % (inspect.stack()[0][3],current_seed))
    plt.xlabel('sensor radius')
    plt.ylabel('time to fully clean')
    plt.show()

def test3(seed=None):
    # set a seed to provide repeatable outcomes each run
    set_seed(seed) # if the seed wasn't set in the input, the default value of none will create (and store) a random seed

    e = NewVacuumEnvironment(width=20,height=20,config="center walls w/ random dirt and fire")
    ef = EnvFrame(e,root=tk.Tk(),cellwidth=30,
                    title='Vacuum Robot Simulation - Scenario=%s(), Seed=%s' % (inspect.stack()[0][3],current_seed))

    # Create agents
    for i in range(1,19):
        e.add_object(GreedyAgentWithRangePerception(sensor_radius = 6), location=(1,i)).id = i

    ef.configure_display()
    ef.run()
    ef.mainloop()

def test4(seed=None):
    # set a seed to provide repeatable outcomes each run
    set_seed(seed) # if the seed wasn't set in the input, the default value of none will create (and store) a random seed

    e = NewVacuumEnvironment(width=20,height=20,config="random dirt")
    ef = EnvFrame(e,root=tk.Tk(),cellwidth=30,
                    title='Vacuum Robot Simulation - Scenario=%s(), Seed=%s' % (inspect.stack()[0][3],current_seed))

    # Create agents on the four corners
    for x in range(2):
        for y in range(2):
            e.add_object(GreedyAgentWithRangePerception(sensor_radius = 6, communication = True), location=(1 + x*(e.width-3),1 + y*(e.height-3))).id = x*2+y+1

    ef.configure_display()
    ef.run()
    ef.mainloop()

def test5(seed=None):
    # set a seed to provide repeatable outcomes each run
    set_seed(seed) # if the seed wasn't set in the input, the default value of none will create (and store) a random seed

    EnvFactory = partial(NewVacuumEnvironment,width=20,height=20,config="random dirt")
    envs = [EnvFactory() for i in range(30)]
    "Return the mean score of running an agent in each of the envs, for steps"
    results = []
    for communication in [True, False]:
        total = 0
        steps = 2000
        i = 0
        for env in copy.deepcopy(envs):
            i+=1
            with Timer(name='Simulation Timer - Comms=%s - Environment=%s' % (communication, i), format='%.4f'):
                for x in range(2):
                    for y in range(2):
                        env.add_object(GreedyAgentWithRangePerception(sensor_radius=6, communication=True),
                                     location=(1 + x * (env.width - 3), 1 + y * (env.height - 3))).id = x*2+y+1
                env.run(steps)
                total += env.t
        results.append(float(total)/len(envs))
    plt.bar(['True', 'False'],[r for r in results],align='center')
    plt.title('scenario=%s(), seed=%s' % (inspect.stack()[0][3],current_seed))
    plt.xlabel('communication')
    plt.ylabel('time to fully clean')
    plt.show()

def test6(seed=None):
    # set a seed to provide repeatable outcomes each run
    set_seed(seed) # if the seed wasn't set in the input, the default value of none will create (and store) a random seed

    e = NewVacuumEnvironment(width=6,height=9,config="shape of eight")
    ef = EnvFrame(e,root=tk.Tk(),cellwidth=30,
                    title='Vacuum Robot Simulation - Scenario=%s(), Seed=%s' % (inspect.stack()[0][3],current_seed))

    # Create agents
    e.add_object(NewGreedyAgentWithRangePerception(sensor_radius = 3, communication = True), location=(1,1)).id = 1

    ef.configure_display()
    ef.run()
    ef.mainloop()

def test7(seed=None):
    # set a seed to provide repeatable outcomes each run
    set_seed(seed) # if the seed wasn't set in the input, the default value of none will create (and store) a random seed

    e = NewVacuumEnvironment(width=50, height=50, config="random dirt")
    ef = EnvFrame(e,root=tk.Tk(), cellwidth=15,
                    title='Vacuum Robot Simulation - Scenario=%s(), Seed=%s' % (inspect.stack()[0][3],current_seed))

    # Create agents
    for i in range(10):
        e.add_object(NewGreedyAgentWithoutRangePerception(communication=True), location=(random.randrange(1,e.width-2), random.randrange(1,e.height-2))).id = i+1

    for i in range(20):
        e.add_object(NewGreedyDrone(sensor_radius=10, communication=True), location=(random.randrange(1,e.width-2), random.randrange(1,e.height-2))).id = i+1

    ef.configure_display()
    ef.run()
    ef.mainloop()

def test8(seed=None):
    # set a seed to provide repeatable outcomes each run
    set_seed(seed) # if the seed wasn't set in the input, the default value of none will create (and store) a random seed

    EnvFactory = partial(NewVacuumEnvironment,width=20,height=20,config="random dirt")
    envs = [EnvFactory() for i in range(10)]
    "Return the mean score of running an agent in each of the envs, for steps"
    results = []
    drone_range = range(0,10,2)
    for num_drones in drone_range:
        total = 0
        steps = 2000
        i = 0
        for env in copy.deepcopy(envs):
            i+=1
            with Timer(name='Simulation Timer - # of Drones=%s - Environment=%s' % (num_drones, i), format='%.4f'):
                for x in range(2):
                    for y in range(2):
                        env.add_object(NewGreedyAgentWithoutRangePerception(communication=True),
                                location=(1 + x * (env.width - 3), 1 + y * (env.height - 3))).id = x * 2 + y + 1

                for n in range(num_drones):
                    env.add_object(NewGreedyDrone(sensor_radius=10, communication=True),
                                 location=(random.randrange(1, 18), random.randrange(1, 18))).id = n + 1

                env.run(steps)
                total += env.t
        results.append(float(total)/len(envs))
    plt.plot(drone_range,[r for r in results],'r-')
    plt.title('scenario=%s(), seed=%s' % (inspect.stack()[0][3],current_seed))
    plt.xlabel('number of drones')
    plt.ylabel('time to fully clean')
    plt.show()

def test9(seed=None):
    # set a seed to provide repeatable outcomes each run
    set_seed(seed) # if the seed wasn't set in the input, the default value of none will create (and store) a random seed

    e = NewVacuumEnvironment(width=20, height=20, config="random dirt")
    ef = EnvFrame(e,root=tk.Tk(), cellwidth=30,
                    title='Vacuum Robot Simulation - Scenario=%s(), Seed=%s' % (inspect.stack()[0][3],current_seed))

    # Create agents
    for i in range(10):
        o = NewGreedyAgentWithRangePerception(sensor_radius=6, communication=True)
        o.actuator_types[3].params['probability'] = .1 if o.actuator_types[3].type is GrabObject else warnings.warn('actuator_type[3] is type %s not type GrabObject' % o.actuator_types[3].type)
        e.add_object(o, location=(random.randrange(1,18), random.randrange(1,18))).id = i+1

    ef.configure_display()
    ef.run()
    ef.mainloop()


def test10(seed=None):
    # set a seed to provide repeatable outcomes each run
    set_seed(seed) # if the seed wasn't set in the input, the default value of none will create (and store) a random seed

    width_max = 40
    height_max = 40

    EnvFactory = partial(NewVacuumEnvironment,width=width_max,height=height_max,config="random dirt")
    envs = [EnvFactory() for i in range(10)]
    "Return the mean score of running an agent in each of the envs, for steps"
    results = []
    agent_range = range(1,3,1)
    for num_agents in agent_range:
        total = 0
        steps = 10000
        i = 0
        for env in copy.deepcopy(envs):
            i+=1
            with Timer(name='Simulation Timer - # of Agents=%s - p=%.3f - Environment=%s' % (num_agents, 1/num_agents, i), format='%.4f'):
                for n in range(num_agents):
                    o = NewGreedyAgentWithRangePerception(sensor_radius=12, communication=True)
                    o.actuator_types[3].params['probability'] = 1/num_agents if o.actuator_types[3].type is GrabObject \
                        else warnings.warn('actuator_type[3] is type %s not type GrabObject' % o.actuator_types[3].type)
                    env.add_object(o, location=(random.randrange(1, width_max-2), random.randrange(1, width_max-2))).id = n + 1

                env.run(steps)
                total += env.t
        results.append(float(total)/len(envs))
    plt.plot(agent_range,[r for r in results],'r-')
    plt.title('scenario=%s(), seed=%s' % (inspect.stack()[0][3],current_seed))
    plt.xlabel('number of agents')
    plt.ylabel('time to fully clean')
    plt.show()

def test11(sensor_radius_min, sensor_radius_max, seed=None, showPlot=True):
    """
    Vary the team makeup (heterogeneity) and communication radius on the drones
    to generate a plot of average completion time vs social entropy vs sensor radius
    """
    # set a seed to provide repeatable outcomes each run
    random.seed(seed) # if the seed wasn't set in the input, the default value of none will create (and store) a random seed

    environment_width = 50
    environment_height = 50
    team_size = 40
    runs_to_average = 100
    max_steps = 3000

    EnvFactory = partial(NewVacuumEnvironment, width=environment_width, height=environment_height, config="random dirt")
    envs = [EnvFactory() for i in range(runs_to_average)]

    # Result lists for plotting should be a list of tuples
    # Every tuple will be structured as follows:
    # (sensor radius [int], ratio of roomba [double], completion times [list])
    data = []

    for sensor_radius in tqdm(range(sensor_radius_min, sensor_radius_max + 1), desc="Sensor radius iterator"):
        for num_drones in tqdm(range(0, team_size), desc="Num drones iterator"):
            num_roomba = team_size - num_drones
            ratio_roomba = num_roomba / team_size
            completion_times = []
            for env in tqdm([EnvFactory() for i in range(runs_to_average)], desc="Runs to avg iterator"):
                for n in range(num_roomba):
                    env.add_object(NewGreedyAgentWithoutRangePerception(communication=True),
                                location=(random.randrange(1, environment_width-2), random.randrange(1, environment_height-2))).id = team_size + n + 1

                for n in range(num_drones):
                    env.add_object(NewGreedyDrone(sensor_radius=sensor_radius, communication=True),
                                 location=(random.randrange(1, environment_width-2), random.randrange(1, environment_width-2))).id = n + 1

                env.run(max_steps)
                completion_times.append(env.t)

            data.append((sensor_radius, ratio_roomba, completion_times))

        # After iterating over all teams, save current data
        pickle.dump(data, open(f"test11_{environment_width}_{environment_height}_{team_size}_{runs_to_average}_{max_steps}_iter{sensor_radius}.p", "wb"))

    # 3D Scatterplot
    if showPlot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sensor_data = [tup[0] for tup in data]
        ratio_data = [tup[1] for tup in data]
        completion_data = [tup[2] for tup in data]
        ax.scatter(sensor_data, ratio_data, [sum(completion_times) / runs_to_average for completion_times in completion_data], marker='o')
        ax.set_title('%s x %s Environment of %s(), seed=%s, team size=%s, agent types=2, averaged over %s runs each' \
                  % (environment_width, environment_height, inspect.stack()[0][3], seed, team_size, runs_to_average))
        ax.set_xlabel('Sensor Radius')
        ax.set_ylabel('Ratio of Roomba')
        ax.set_zlabel('Average Completion Time')
        plt.show()

def test13(seed=None):
    # set a seed to provide repeatable outcomes each run
    set_seed(seed) # if the seed wasn't set in the input, the default value of none will create (and store) a random seed

    e = NewVacuumEnvironment(width=20, height=20, config="random dirt")
    ef = EnvFrame(e,root=tk.Tk(), cellwidth=30,
                    title='Vacuum Robot Simulation - Scenario=%s(), Seed=%s' % (inspect.stack()[0][3],current_seed))

    # Create agents
    for i in range(10):
        o = NewKMeansAgentWithNetworkComms(sensor_radius=5, comms_range=5)
        e.add_object(o, location=(random.randrange(1,18), random.randrange(1,18))).id = i+1

    ef.configure_display()
    ef.run()
    ef.mainloop()


def test14(seed=None):
    # set a seed to provide repeatable outcomes each run
    set_seed(seed) # if the seed wasn't set in the input, the default value of none will create (and store) a random seed

    repetitions = 10

    width_max = 20
    height_max = 20

    EnvFactory = partial(NewVacuumEnvironment,width=width_max,height=height_max,config="random dirt")
    envs = [EnvFactory() for i in range(repetitions)]
    # Return the mean score of running an agent in each of the envs, for steps
    results = []
    num_agents = 10
    comms_range = range(1,11,1)
    sensor_range = range(1,11,1)
    results = np.zeros((len(comms_range),len(sensor_range)))
    for i, cr in enumerate(comms_range):
        for j, sr in enumerate(sensor_range):
            total = 0
            steps = 1000
            count = 0
            for env in copy.deepcopy(envs):
                count += 1
                with Timer(name='Simulation Timer - Comms Range=%s - Sensor Range=%s - Environment=%s' % (cr, sr, count), format='%.4f'):
                    for n in range(num_agents):
                        o = NewKMeansAgentWithNetworkComms(sensor_radius=sr, comms_range=cr)
                        env.add_object(o, location=(
                            random.randrange(1, width_max - 2), random.randrange(1, width_max - 2))).id = n + 1

                    env.run(steps)
                    total += env.t
            results[i,j] = float(total)/len(envs)

    print(results)

    X,Y = np.meshgrid(comms_range, sensor_range)
    f = plt.figure()
    ax = f.gca(projection = '3d')

    surf = ax.plot_surface(X, Y, np.transpose(results), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)



    plt.title('scenario=%s(), seed=%s, (%sx%s) w/ %s roombas' % (inspect.stack()[0][3],current_seed, width_max, height_max, num_agents))
    ax.set_xlabel('comms range')
    ax.set_ylabel('sensor range')
    ax.set_zlabel('time to clean')
    plt.show()


def test15(seed=None):
    # set a seed to provide repeatable outcomes each run
    set_seed(seed) # if the seed wasn't set in the input, the default value of none will create (and store) a random seed

    repetitions = 10
    width_max = 50
    height_max = 50
    num_agents = 10

    EnvFactory = partial(NewVacuumEnvironment,width=width_max,height=height_max,config="random dirt")
    envs = [EnvFactory() for i in range(repetitions)]
    "Return the mean score of running an agent in each of the envs, for steps"
    results = []
    confidence_intervals = []
    sr = 10
    comms_range = range(10,51,2)
    for cr in comms_range:
        total = []
        steps = 1000
        i = 0
        for env in copy.deepcopy(envs):
            i+=1
            with Timer(name='Simulation Timer - Comms Range=%s - Environment=%s' % (cr, i), format='%.4f'):
                for n in range(num_agents):
                    o = NewKMeansAgentWithNetworkComms(sensor_radius=sr, comms_range=cr)
                    env.add_object(o, location=(random.randrange(1, width_max-2), random.randrange(1, width_max-2))).id = n + 1

                env.run(steps)
                total.append(env.t)
        results.append(float(sum(total))/len(envs))
        _, start, end = confidence_interval(total) # 95% CI by default
        confidence_intervals.append((end-start) / 2)
    plt.errorbar(comms_range,[r for r in results], yerr=confidence_intervals, label='95% confidence interval')
    plt.title('scenario=%s(), seed=%s, sensor radius = %s, (%sx%s) w/ %s roombas' % (inspect.stack()[0][3],current_seed, sr, width_max, height_max, num_agents))
    plt.xlabel('comms range')
    plt.ylabel('time to fully clean')
    plt.show()


def test16(seed=None):
    # set a seed to provide repeatable outcomes each run
    set_seed(seed) # if the seed wasn't set in the input, the default value of none will create (and store) a random seed

    kmeans = True  # change this to toggle between k-means and greedy agent behavior

    e = NewVacuumEnvironment(width=50, height=50, config="corner dirt")
    ef = EnvFrame(e,root=tk.Tk(), cellwidth=18,
                    title='Vacuum Robot Simulation - Scenario=%s(), Seed=%s' % (inspect.stack()[0][3],current_seed))

    # Create agents
    i = 0
    for (x,y) in [(14,14),(14,24),(24,14),(24,24)]:
        i += 1
        if kmeans:
            o = NewKMeansAgentWithNetworkComms(sensor_radius=10, comms_range=30)
        else:
            o = NewGreedyAgentWithRangePerception(sensor_radius=100, communication=True)
        e.add_object(o, location=(x,y)).id = i

    e.add_object(Dirt(), location=(24,25))

    ef.configure_display()
    ef.run(pause=True)
    ef.mainloop()


def test17(seed=None, kmeans=True):
    # set a seed to provide repeatable outcomes each run
    set_seed(seed) # if the seed wasn't set in the input, the default value of none will create (and store) a random seed

    e = NewVacuumEnvironment(width=50, height=50, config="empty")
    ef = EnvFrame(e,root=tk.Tk(), cellwidth=15,
                    title='Vacuum Robot Simulation - Scenario=%s(), Seed=%s' % (inspect.stack()[0][3],current_seed))

    # Create agents
    i = 0
    for (x,y) in [(20,20),(20,21),(21,20),(21,21)]:
        i += 1
        if kmeans:
            o = NewKMeansAgentWithNetworkComms(sensor_radius=100, comms_range=100)
        else:
            o = NewGreedyAgentWithRangePerception(sensor_radius=100, communication=True)
        e.add_object(o, location=(x,y)).id = i

    e.add_object(Dirt(), location=(1,1))
    e.add_object(Dirt(), location=(44,1))
    e.add_object(Dirt(), location=(1,48))
    e.add_object(Dirt(), location=(48,48))

    ef.configure_display()
    ef.run(pause=True)
    ef.mainloop()


def test18(seed=None, kmeans=True):
    # set a seed to provide repeatable outcomes each run
    set_seed(seed) # if the seed wasn't set in the input, the default value of none will create (and store) a random seed

    e = NewVacuumEnvironment(width=50, height=50, config="empty")
    ef = EnvFrame(e,root=tk.Tk(), cellwidth=15,
                    title='Vacuum Robot Simulation - Scenario=%s(), Seed=%s' % (inspect.stack()[0][3],current_seed))

    # Create agents
    i = 0
    for (x,y) in [(20,20),(20,21),(21,20),(21,21)]:
        i += 1
        if kmeans:
            o = NewKMeansAgentWithNetworkComms(sensor_radius=100, comms_range=100)
        else:
            o = NewGreedyAgentWithRangePerception(sensor_radius=100, communication=True)
        e.add_object(o, location=(x,y)).id = i

    # create the 'problem'
    e.problem = Problem(e,n=50,type='acyclic')

    for node in e.problem.graph:
        d = Dirt(id=node)
        e.add_object(d, location=(random.randrange(1,e.width-1), random.randrange(1,e.height-1)))

    ef.configure_display()
    ef.run(pause=False)
    ef.mainloop()


def test_all(seed=None):
    test0(seed)
    test1(seed)
    test2(seed)
    test3(seed)
    test4(seed)
    test5(seed)
    test6(seed)
    test7(seed)
    test8(seed)
    test9(seed)
    test10(seed)

def main():
    test11()
    #test_all()

if __name__ == "__main__":
    # execute only if run as a script
    main()
