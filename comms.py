
from agents import Agent


class Communicator():
    def __init__(self, env):
        self.env = env

    def get_comms_network(self, toAgent):
        '''return the list of Agents that the toAgent is able to communicate with'''
        return None

    def communicate(self, message, toAgent, fromAgent):
        '''communicate a message from the fromAgent to the toAgent'''
        return None

    def run_comms(self, agents):
        for to_a in agents:
            network = self.get_comms_network(to_a) # this could be moved to the line below, but leaving it as two lines for readability
            a.messages = [self.communicate('percept', to_a, from_a) for from_a in network]

class BroadcastCommunicator(Communicator):

    def get_comms_network(self, toAgent):
        range = 5
        return [o for o in self.env.objects_near(toAgent.location, range) if isinstance(o, Agent)]

    def communicate(self, message, toAgent, fromAgent):
        pass