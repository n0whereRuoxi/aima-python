'''
This file holds the agents.

'''

import random, copy, collections, math
from objects import Object
from perception import *
from comms import *
from actuator import *
import numpy as np
from scipy import spatial
from utils import distance2, unit_vector, vector_add, scalar_vector_product, vector_average
import uuid
# ______________________________________________________________________________

class Agent(Object):
    '''
    An Agent is a subclass of Object with one required slot, .program, which should hold a function that takes one
    argument, the percept, and returns an action. (What counts as a percept or action will depend on the specific
    environment in which the agent exists.)  Note that 'program' is a slot, not a method.  If it were a method, then
    the program could 'cheat' and look at aspects of the agent.  It's not supposed to do that: the program can only
    look at the percepts.  An agent program that needs a model of the world (and of the agent itself) will have to
    build and maintain its own model.  There is an optional slots, .performance, which is a number giving the
    performance measure of the agent in its environment.
    '''
    def __init__(self):
        def program(percept):
            return raw_input('Percept=%s; action? ' % percept)

        self.program = program
        self.alive = True
        self.perceptor_types = [BasicPerceptor]
        self.holding = []
        self.current_task = ('None',0) # task and current duration

        self.performance = 0

        if program is None or not isinstance(program, collections.Callable):
            print("Can't find a valid program for {}, falling back to default.".format(
                self.__class__.__name__))

            def program(percept):
                return eval(input('Percept={}; action? '.format(percept)))

    blocker = True

def DebugAgent(agent):
    '''
    Wrap the agent's program to print its input and output. This will let you see what the agent is doing in the
    environment.
    '''

    # Mark: Do we just replace the agent parent class with DebugAgent to enable printing?
    old_program = agent.program
    def new_program(percept):
        action = old_program(percept)
        print('%s perceives %s and does %s' % (agent, percept, action))
        return action
    agent.program = new_program
    return agent
#______________________________________________________________________________

class XYAgent(Agent):
    holding = []
    heading = (1, 0)
    actuator_types = [Component(type=MoveForward), Component(type=TurnLeft), Component(type=TurnRight)]

class RandomXYAgent(XYAgent):
    "An agent that chooses an action at random, ignoring all percepts."

    def __init__(self, actions):
        Agent.__init__(self)
        self.program = lambda percept: random.choice(actions)

def NewRandomXYAgent(debug=False):
    "Randomly choose one of the actions from the vaccum environment."
    # the extra forwards are just to alter the probabilities
    if debug:
        return DebugAgent(RandomXYAgent(['TurnRight', 'TurnLeft', 'MoveForward', 'MoveForward', 'MoveForward', 'MoveForward', 'MoveForward', 'MoveForward']))
    else:
        return RandomXYAgent(['TurnRight', 'TurnLeft', 'MoveForward', 'MoveForward', 'MoveForward', 'MoveForward', 'MoveForward', 'MoveForward'])
    #return RandomXYAgent(['TurnRight', 'TurnLeft', 'MoveForward', 'GrabObject', 'ReleaseObject'])

class RandomReflexAgent(XYAgent):
    '''This agent takes action based solely on the percept. [Fig. 2.13]'''
    def __init__(self, actions):
        Agent.__init__(self)
        self.actions = actions
        self.perceptor_types = [DirtyPerceptor, BumpPerceptor]
        self.actuator_types.extend([Component(type=GrabObject), Component(type=ReleaseObject)])
        def program(percept):
            if percept['Dirty']:
                return "GrabObject"
            elif percept['Bump']:
                return random.choice(['TurnRight','TurnLeft'])
            else:
                return random.choice(actions)
        self.program = program

def find_nearest(agent_location, dirts):
    if len(dirts) == 1:
        return dirts[0]
    return dirts[spatial.KDTree(np.asarray(dirts)).query(np.asarray(agent_location))[1]]

def go_to(agent_location, agent_heading, nearest_dirt, bump):
    if agent_heading[0] == 0:
        '''up or down'''
        if (nearest_dirt[1] - agent_location[1]) * agent_heading[1] > 0 and not bump:
            return 'MoveForward'
        else:
            if nearest_dirt[0] - agent_location[0] > 0:
                '''dirt to right'''
                if agent_heading[1] == 1:
                    return 'TurnRight'
                else:
                    return 'TurnLeft'
            else:
                if agent_heading[1] == 1:
                    return 'TurnLeft'
                else:
                    return 'TurnRight'
    else:
        '''left or right'''
        if (nearest_dirt[0] - agent_location[0]) * agent_heading[0] > 0 and not bump:
            return 'MoveForward'
        else:
            if nearest_dirt[1] - agent_location[1] > 0:
                '''dirt to down'''
                if agent_heading[0] == 1:
                    return 'TurnLeft'
                else:
                    return 'TurnRight'
            else:
                if agent_heading[0] == 1:
                    return 'TurnRight'
                else:
                    return 'TurnLeft'

def basic_state_estimator(percepts, comms):
    if not 'Objects' in percepts: percepts['Objects'] = [] # if percepts['Objects'] is empty, initialize it as an empty list
    for comm in comms.values():
        if 'Objects' in comm: percepts['Objects'] += [(o[0], (  # o[1] is the location tuple, and o[1][0] is the x and o[1][1] is the y
                        o[1][0] + comm['GPS'][0] - percepts['GPS'][0], o[1][1] + comm['GPS'][1] - percepts['GPS'][1])) for
                        o in comm['Objects']]

    # convert dirts to a set to remove duplicates and convert back to a list
    percepts['Objects'] = list(set(percepts['Objects']))  # note: does not preserve order

    for o in percepts['Objects']:
        if o[1][0] > 19 or o[1][1] > 19:
            print(percepts['GPS'], o)

    return percepts

class GreedyAgentWithRangePerception(XYAgent):
    '''This agent takes action based solely on the percept. [Fig. 2.13]'''

    def __init__(self, sensor_radius=10, communication=False):
        Agent.__init__(self)
        self.perceptor_types = [GPSPerceptor, DirtyPerceptor, BumpPerceptor, CompassPerceptor, RangePerceptor]
        self.actuator_types.extend([Component(type=GrabObject), Component(type=ReleaseObject)])
        self.communicator = Communicator if communication else None
        self.sensor_r = sensor_radius
        self.comms = {}
        # orientation = {(1,0): 'right', (-1,0): 'left', (0,-1): 'up', (0,1): 'down'}
        # def turn_heading(heading, inc, headings=[(1, 0), (0, 1), (-1, 0), (0, -1)]):
        #     "Return the heading to the left (inc=+1) or right (inc=-1) in headings."
        #     return headings[(headings.index(heading) + inc) % len(headings)]
        self.dirts = []
        def program(percepts):
            if percepts['Dirty']:
                return 'GrabObject'
            else:
                agent_location = (0,0)
                agent_heading = percepts['Compass']
                # collect communication data
                dirts = [o[1] for o in percepts['Objects'] if o[0] == 'Dirt']

                # convert dirts to a set to remove duplicates and convert back to a list
                dirts = list(set(dirts))  # note: does not preserve order

                if dirts:
                    self.dirts = dirts
                    nearest_dirt = find_nearest(agent_location, dirts)
                    command = go_to(agent_location, agent_heading, nearest_dirt, percepts['Bump'])
                    return command
                return random.choice(['TurnRight', 'TurnLeft', 'MoveForward', 'MoveForward', 'MoveForward', 'MoveForward'])
        self.program = program

        self.state_estimator = basic_state_estimator

def NewGreedyAgentWithRangePerception(debug=False, sensor_radius=10, communication=False):
    "Randomly choose one of the actions from the vaccum environment."
    # the extra forwards are just to alter the probabilities
    if debug:
        return DebugAgent(GreedyAgentWithRangePerception(sensor_radius=sensor_radius))
    else:
        return GreedyAgentWithRangePerception(sensor_radius=sensor_radius)

class GreedyAgentWithoutRangePerception(XYAgent):
    '''
    This agent is designed to communicate with drones to identify where the dirt is.
    It doesn't have a RangePerceptor so it must commmunicate.
    '''

    def __init__(self, communication=True):
        Agent.__init__(self)
        self.perceptor_types = [GPSPerceptor, DirtyPerceptor, BumpPerceptor, CompassPerceptor]
        self.actuator_types.extend([Component(type=GrabObject,params={'probability':1}), Component(type=ReleaseObject)])
        self.communicator = Communicator if communication else None
        self.comms = {}
        # orientation = {(1,0): 'right', (-1,0): 'left', (0,-1): 'up', (0,1): 'down'}
        self.dirts = []
        def program(percepts):
            if percepts['Dirty']:
                return 'GrabObject'
            else:
                agent_location = (0,0)
                agent_heading = percepts['Compass']
                # collect communication data
                dirts = [o[1] for o in percepts['Objects'] if o[0] == 'Dirt']

                if dirts:
                    self.dirts = dirts
                    nearest_dirt = find_nearest(agent_location, dirts)
                    command = go_to(agent_location, agent_heading, nearest_dirt, percepts['Bump'])
                    return command
                return random.choice(['TurnRight', 'TurnLeft', 'MoveForward', 'MoveForward', 'MoveForward', 'MoveForward'])
        self.program = program

        self.state_estimator = basic_state_estimator

def NewGreedyAgentWithoutRangePerception(debug=False, communication=True):
    "Randomly choose one of the actions from the vaccum environment."
    # the extra forwards are just to alter the probabilities
    if debug:
        return DebugAgent(GreedyAgentWithoutRangePerception(communication=communication))
    else:
        return GreedyAgentWithoutRangePerception(communication=communication)

class GreedyDrone(XYAgent):
    '''
    This agent is designed to communicate with drones to identify where the dirt is.
    It doesn't have a RangePerceptor so it must commmunicate.
    '''

    blocker = False

    def __init__(self, sensor_radius=10, communication=True):
        Agent.__init__(self)
        self.perceptor_types = [GPSPerceptor, CompassPerceptor, RangePerceptor]
        self.actuator_types.extend([]) # No additional actions, just turn and move
        self.communicator = Communicator if communication else None
        self.comms = {}
        # orientation = {(1,0): 'right', (-1,0): 'left', (0,-1): 'up', (0,1): 'down'}
        self.dirts = []
        self.sensor_r = sensor_radius
        def program(percepts):
            agent_location = (0,0)
            agent_heading = percepts['Compass']
            # collect communication data
            dirts = [o[1] for o in percepts['Objects'] if o[0] == 'Dirt']
            drones = [d[1] for d in percepts['Objects'] if d[0] == 'GreedyDrone']

            if dirts:
                self.dirts = dirts

                close_dirts = [d for d in dirts if distance2(d,agent_location)<(sensor_radius*.75)**2]
                if close_dirts: # if there are dirts close to you, move towards the center (of mass) of them
                    target = vector_average(close_dirts)
                else: # if there are no dirts close to you, move towards the closest dirt
                    target = find_nearest(agent_location, dirts)

                if drones:  # if there are drones around, move away from them by half your sensor radius
                    targets = [target]
                    for d in [d for d in drones if distance2(d, agent_location) < (sensor_radius * .5) ** 2]:
                        targets.append(vector_add(scalar_vector_product(
                            sensor_radius * .5, vector_add(agent_location, scalar_vector_product(-1, d))),
                                                  agent_location))
                    target = vector_average(targets)

                command = go_to(agent_location, agent_heading, target, bump=False)
                return command
            elif drones: # if no dirts, but there are drones around
                targets = []
                for d in [d for d in drones if distance2(d,agent_location)<(sensor_radius*.5)**2]:
                    targets.append(vector_add(scalar_vector_product(sensor_radius*.5, vector_add(agent_location,scalar_vector_product(-1,d))),agent_location))
                if targets:
                    target = vector_average(targets)
                    return go_to(agent_location, agent_heading, target, bump=False)
                else:
                    return random.choice(
                        ['TurnRight', 'TurnLeft', 'MoveForward', 'MoveForward', 'MoveForward', 'MoveForward'])
            else:  # if no dirts and no drones, make a random action
                return random.choice(['TurnRight', 'TurnLeft', 'MoveForward', 'MoveForward', 'MoveForward', 'MoveForward'])
        self.program = program

        self.state_estimator = basic_state_estimator

def NewGreedyDrone(debug=False, sensor_radius=10, communication=True):
    "Randomly choose one of the actions from the vaccum environment."
    # the extra forwards are just to alter the probabilities
    if debug:
        return DebugAgent(GreedyDrone(sensor_radius=sensor_radius, communication=communication))
    else:
        return GreedyDrone(sensor_radius=sensor_radius, communication=communication)


class GreedyAgent(XYAgent):
    '''This agent takes action based solely on the percept. [Fig. 2.13]'''
    def __init__(self):
        Agent.__init__(self)
        # orientation = {(1,0): 'right', (-1,0): 'left', (0,-1): 'up', (0,1): 'down'}
        # def turn_heading(heading, inc, headings=[(1, 0), (0, 1), (-1, 0), (0, -1)]):
        #     "Return the heading to the left (inc=+1) or right (inc=-1) in headings."
        #     return headings[(headings.index(heading) + inc) % len(headings)]
        self.perceptor_types = [DirtyPerceptor, BumpPerceptor, GPSPerceptor, CompassPerceptor, PerfectPerceptor]
        self.actuator_types.extend([Component(type=GrabObject), Component(type=ReleaseObject)])
        def program(percepts):
            if percepts['Dirty']:
                return "Grab"
            else:
                dirts = [o[1] for o in percepts['Objects'] if o[0]=='Dirt']
                agent_location = percepts['GPS']
                agent_heading = percepts['Compass']
                if dirts:
                    nearest_dirt = find_nearest(agent_location, dirts)
                    command = go_to(agent_location, agent_heading, nearest_dirt, percepts['Bump'])
                    return command
                return ''
        self.program = program


def NewRandomReflexAgent(debug=False):
    "If the cell is dirty, Grab the dirt; otherwise, randomly choose one of the actions from the vaccum environment."
    # the extra forwards are just to alter the probabilities
    if debug:
        return DebugAgent(RandomReflexAgent(['TurnRight', 'TurnLeft', 'MoveForward', 'MoveForward', 'MoveForward', 'MoveForward', 'MoveForward', 'MoveForward']))
    else:
        return RandomReflexAgent(['TurnRight', 'TurnLeft', 'MoveForward', 'MoveForward', 'MoveForward', 'MoveForward', 'MoveForward', 'MoveForward'])

class SimpleReflexAgent(XYAgent):
    '''This agent takes action based solely on the percept. [Fig. 2.13]'''

    def __init__(self, rules, interpret_input):
        Agent.__init__(self)
        def program(percept):
            state = interpret_input(percept)
            rule = rule_match(state, rules)
            action = rule.action
            return action
        self.program = program

class ReflexAgentWithState(XYAgent):
    '''This agent takes action based on the percept and state. [Fig. 2.16]'''

    def __init__(self, rules, udpate_state):
        Agent.__init__(self)
        state, action = None, None
        def program(percept):
            state = update_state(state, action, percept)
            rule = rule_match(state, rules)
            action = rule.action
            return action
        self.program = program

#______________________________________________________________________________
