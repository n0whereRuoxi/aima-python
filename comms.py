import agents
import networkx as nx
from utils import distance2

class Communicator():
    def __init__(self, env):
        self.env = env


    def setup(self):
        for a in self.env.agents:
            a.comms = {}


    def get_comms_network(self, from_agent):
        '''return the list of Agents that the from_agent is able to communicate with'''
        return [a for a in self.env.agents if not a is from_agent]


    def communicate(self, message, from_agent, to_agent):
        '''communicate a message from the from_agent to the to_agent'''
        to_agent.comms[from_agent.id] = message


class BroadcastCommunicator(Communicator):
    def __init__(self, env, range=5):
        Communicator.__init__(self, env)
        self.range = range


    def get_comms_network(self, to_agent):
        range = 5
        return [o for o in self.env.objects_near(to_agent.location, range) if isinstance(o, agents.Agent) and to_agent != o]


class NetworkCommunicator(Communicator):
    def __init__(self, env, range=5):
        Communicator.__init__(self, env)
        self.range = range

        self.network = nx.Graph()


    def setup(self):
        Communicator.setup(self)
        self.build_network()


    def get_comms_network(self, from_agent):
        return [o for o in self.env.objects_near(from_agent.location, self.range)
                if isinstance(o, agents.Agent) and from_agent != o and nx.connectivity.node_connectivity(self.network, from_agent, o)]


    def build_network(self):
        self.network = nx.Graph()
        for u in self.env.agents:
            for v in [a for a in self.env.agents if distance2(u.location,a.location) <= self.range**2]:
                if not u is v: self.network.add_edge(u,v)

