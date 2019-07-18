'''
This file holds the agents.

'''

import random, copy, collections
from objects import Object
from perception import *
from comms import *
import numpy as np
from scipy import spatial
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
        self.perceptorTypes = [BasicPerceptor]
        self.holding = []

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

class RandomXYAgent(XYAgent):
    "An agent that chooses an action at random, ignoring all percepts."

    def __init__(self, actions):
        Agent.__init__(self)
        self.program = lambda percept: random.choice(actions)

def NewRandomXYAgent(debug=False):
    "Randomly choose one of the actions from the vaccum environment."
    # the extra forwards are just to alter the probabilities
    if debug:
        return DebugAgent(RandomXYAgent(['TurnRight', 'TurnLeft', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward']))
    else:
        return RandomXYAgent(['TurnRight', 'TurnLeft', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward'])
    #return RandomXYAgent(['TurnRight', 'TurnLeft', 'Forward', 'Grab', 'Release'])

class RandomReflexAgent(XYAgent):
    '''This agent takes action based solely on the percept. [Fig. 2.13]'''
    def __init__(self, actions):
        Agent.__init__(self)
        self.actions = actions
        self.perceptorTypes = [DirtyPerceptor, BumpPerceptor]
        def program(percept):
            if percept['Dirty']:
                return "Grab"
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
            return 'Forward'
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
            return 'Forward'
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

class GreedyAgentWithRangePerception(XYAgent):
    '''This agent takes action based solely on the percept. [Fig. 2.13]'''

    def __init__(self, sensor_radius=10, communication=False):
        Agent.__init__(self)
        self.perceptorTypes = [GPSPerceptor, DirtyPerceptor, BumpPerceptor, CompassPerceptor, RangePerceptor]
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
                return 'Grab'
            else:
                # collect percept data
                dirts = [o[1] for o in percepts['Objects'] if o[0]=='Dirt']
                agent_location = (0, 0)
                agent_heading = percepts['Compass']
                # collect communication data
                for comm in self.comms.values():
                    dirts += [(o[1][0] + comm['GPS'][0] - percepts['GPS'][0], o[1][1] + comm['GPS'][1] - percepts['GPS'][1]) for o in comm['Objects'] if o[0] == 'Dirt']
                if dirts:
                    self.dirts = dirts
                    nearest_dirt = find_nearest(agent_location, dirts)
                    command = go_to(agent_location, agent_heading, nearest_dirt, percepts['Bump'])
                    return command
                return random.choice(['TurnRight', 'TurnLeft', 'Forward', 'Forward', 'Forward', 'Forward'])
        self.program = program

def NewGreedyAgentWithRangePerception(debug=False, sensor_radius=10):
    "Randomly choose one of the actions from the vaccum environment."
    # the extra forwards are just to alter the probabilities
    if debug:
        return DebugAgent(GreedyAgentWithRangePerception(sensor_radius=sensor_radius))
    else:
        return GreedyAgentWithRangePerception(sensor_radius=sensor_radius)


class GreedyAgent(XYAgent):
    '''This agent takes action based solely on the percept. [Fig. 2.13]'''
    def __init__(self):
        Agent.__init__(self)
        # orientation = {(1,0): 'right', (-1,0): 'left', (0,-1): 'up', (0,1): 'down'}
        # def turn_heading(heading, inc, headings=[(1, 0), (0, 1), (-1, 0), (0, -1)]):
        #     "Return the heading to the left (inc=+1) or right (inc=-1) in headings."
        #     return headings[(headings.index(heading) + inc) % len(headings)]
        self.perceptorTypes = [DirtyPerceptor, BumpPerceptor, GPSPerceptor, CompassPerceptor, PerfectPerceptor]
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
        return DebugAgent(RandomReflexAgent(['TurnRight', 'TurnLeft', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward']))
    else:
        return RandomReflexAgent(['TurnRight', 'TurnLeft', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward'])

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
