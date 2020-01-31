'''
This file holds the agents.

'''

# import external code files
from objects import Object
from perception import *
from comms import *
from actuator import *
import programs

import random, copy, collections, math
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
    def __init__(self):
        Agent.__init__(self)
        self.holding = []
        self.heading = (1, 0)
        self.actuator_types = [Component(type=MoveForward), Component(type=TurnLeft), Component(type=TurnRight)]

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
        self.program = programs.random_reflex_generator(actions)


class GreedyAgentWithRangePerception(XYAgent):
    '''This agent takes action based solely on the percept. [Fig. 2.13]'''

    def __init__(self, sensor_radius=10, communication=False):
        XYAgent.__init__(self)
        self.perceptor_types = [GPSPerceptor, DirtyPerceptor, BumpPerceptor, CompassPerceptor, RangePerceptor]
        self.actuator_types.extend([Component(type=GrabObject), Component(type=ReleaseObject)])
        self.communicator = Communicator if communication else None
        self.sensor_r = sensor_radius
        self.comms = {}

        self.program = programs.greedy_roomba_generator()
        self.state_estimator = programs.basic_state_estimator_generator()


def NewGreedyAgentWithRangePerception(debug=False, sensor_radius=10, communication=False):
    "Randomly choose one of the actions from the vaccum environment."
    # the extra forwards are just to alter the probabilities
    if debug:
        return DebugAgent(GreedyAgentWithRangePerception(sensor_radius=sensor_radius))
    else:
        return GreedyAgentWithRangePerception(sensor_radius=sensor_radius, communication=communication)

class GreedyAgentWithoutRangePerception(XYAgent):
    '''
    This agent is designed to communicate with drones to identify where the dirt is.
    It doesn't have a RangePerceptor so it must commmunicate.
    '''

    def __init__(self, communication=True):
        XYAgent.__init__(self)
        self.perceptor_types = [GPSPerceptor, DirtyPerceptor, BumpPerceptor, CompassPerceptor]
        self.actuator_types.extend([Component(type=GrabObject,params={'probability':1}), Component(type=ReleaseObject)])
        self.communicator = Communicator if communication else None
        self.comms = {}
        self.program = programs.greedy_roomba_generator()

        self.state_estimator = programs.basic_state_estimator_generator()

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
        XYAgent.__init__(self)
        self.perceptor_types = [GPSPerceptor, CompassPerceptor, RangePerceptor]
        self.actuator_types.extend([]) # No additional actions, just turn and move
        self.communicator = Communicator if communication else None
        self.comms = {}
        self.sensor_r = sensor_radius

        self.program = programs.greedy_drone_generator(sensor_radius)
        self.state_estimator = programs.basic_state_estimator_generator()

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
        self.program = programs.greedy_roomba_generator()


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
        self.program = programs.rule_program_generator()


class ReflexAgentWithState(XYAgent):
    '''This agent takes action based on the percept and state. [Fig. 2.16]'''

    def __init__(self, rules, udpate_state):
        Agent.__init__(self)
        state, action = None, None
        self.program = programs.rule_program_generator()


class KMeansAgentWithNetworkComms(XYAgent):
    '''This agent takes action based solely on the percept. [Fig. 2.13]'''

    def __init__(self, sensor_radius=10, comms_range=5):
        XYAgent.__init__(self)
        self.perceptor_types = [GPSPerceptor, DirtyPerceptor, BumpPerceptor, CompassPerceptor, RangePerceptor]
        self.actuator_types.extend([Component(type=GrabObject), Component(type=ReleaseObject)])
        #self.communicator = BroadcastCommunicator if comms_range>=1 else None
        self.communicator = NetworkCommunicator if comms_range>=1 else None
        self.sensor_r = sensor_radius
        self.comms_r = comms_range
        self.comms = {}

        self.program = programs.kmeans_roomba_generator()
        self.state_estimator = programs.basic_state_estimator_generator()


def NewKMeansAgentWithNetworkComms(debug=False, sensor_radius=10, comms_range=5):
    "Randomly choose one of the actions from the vaccum environment."
    # the extra forwards are just to alter the probabilities
    if debug:
        return DebugAgent(KMeansAgentWithNetworkComms(sensor_radius=sensor_radius, comms_range=comms_range))
    else:
        return KMeansAgentWithNetworkComms(sensor_radius=sensor_radius, comms_range=comms_range)


def NewColorKMeansAgentWithNetworkComms(debug=False, sensor_radius=10, comms_range=5, color='red'):
    "Randomly choose one of the actions from the vaccum environment."
    # the extra forwards are just to alter the probabilities
    ag = KMeansAgentWithNetworkComms(sensor_radius=sensor_radius, comms_range=comms_range)
    ag.program = programs.a
    if debug:
        return DebugAgent(ag)
    else:
        return ag


class GraphAgent(XYAgent):
    '''This agent takes action based solely on the percept. [Fig. 2.13]'''

    def __init__(self, comms_range, sensor_radius):
        XYAgent.__init__(self)
        self.perceptor_types = [GPSPerceptor, DirtyPerceptor, BumpPerceptor, CompassPerceptor, RangePerceptor, DirtsCleanedPerceptor]
        self.actuator_types.extend([Component(type=GrabObject), Component(type=ReleaseObject)])
        self.communicator = NetworkCommunicator if comms_range >= 1 else None
        self.sensor_r = sensor_radius
        self.comms_r = comms_range
        self.comms = {}
        self.state = {}

        self.program = programs.dag_roomba_generator()
        self.state_estimator = programs.graph_state_estimator_generator()


def NewGraphAgent(debug=False, sensor_radius=10, comms_range=5):
    ag = GraphAgent(sensor_radius=sensor_radius, comms_range=comms_range)
    ag.program = programs.a
    if debug:
        return DebugAgent(ag)
    else:
        return ag

#______________________________________________________________________________
