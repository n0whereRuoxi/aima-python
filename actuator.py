from utils import vector_add
from objects import Dirt
import random

class Actuator():
    def __init__(self, env):
        self.env = env

    def actuate(self, agent, params=None):
        ''' This is the code that runs the behavior on the environment.  Needs to be overwritten'''
        pass

class MoveForward(Actuator):
    def actuate(self, agent, params=None):
        # move the Agent in the facing direction by adding the heading vector to the Agent location
        if random.random() <= params.get('probability',1):
            self.env.move_to(agent, vector_add(agent.heading, agent.location))

def turn_heading(heading, inc,
             headings=[(1, 0), (0, 1), (-1, 0), (0, -1)]):
    "Return the heading to the left (inc=+1) or right (inc=-1) in headings."
    return headings[(headings.index(heading) + inc) % len(headings)]

class TurnRight(Actuator):
    def actuate(self, agent, params=None):
        # decrement the heading by -90° by getting the previous index of the headings array
        if random.random() <= params.get('probability',1):
            agent.heading = turn_heading(agent.heading, -1)

class TurnLeft(Actuator):
    def actuate(self, agent, params=None):
        # increment the heading by +90° by getting the next index of the headings array
        if random.random() <= params.get('probability',1):
            agent.heading = turn_heading(agent.heading, +1)

class GrabObject(Actuator):
    def actuate(self, agent, params=None):
        # check to see if any objects at the Agent's location are grabbable by the Agent
        if random.random() <= params.get('probability',1):
            objs = [obj for obj in self.env.objects_at(agent.location) if (obj != agent and obj.is_grabbable(agent))]
            # if so, pick up all grabbable objects and add them to the holding array
            if objs:
                agent.holding += objs
                for o in objs:
                    # set the location of the Object = the Agent instance carrying the Object
                    # by setting the location to an object instead of a tuple, we can now detect
                    # when to remove if from the display.  This may be useful in other ways, if
                    # the object needs to know who it's holder is
                    o.location = agent
                    if isinstance(o,Dirt): agent.performance += 100

class ReleaseObject(Actuator):
    def actuate(self, agent, params=None):
        if random.random() <= params.get('probability',1):
            # drop an objects being held by the Agent.
            if agent.holding:
                # restore the location parameter to add the object back to the display
                agent.holding.pop().location = agent.location
