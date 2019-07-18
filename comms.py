import agents

class Communicator():
    def __init__(self, env):
        self.env = env

    def get_comms_network(self, from_agent):
        '''return the list of Agents that the from_agent is able to communicate with'''
        return self.env.agents

    def communicate(self, message, from_agent, to_agent):
        '''communicate a message from the from_agent to the to_agent'''
        to_agent.comms[from_agent.id] = message
        return None

    def run_comms(self, agents):
        for to_a in agents:
            network = self.get_comms_network(to_a) # this could be moved to the line below, but leaving it as two lines for readability
            a.messages = [self.communicate('percept', to_a, from_a) for from_a in network]

class BroadcastCommunicator(Communicator):

    def get_comms_network(self, to_agent):
        range = 5
        return [o for o in self.env.objects_near(to_agent.location, range) if isinstance(o, agents.Agent)]

    def communicate(self, message, to_agent, from_agent):
        pass
