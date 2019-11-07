# define the list of programs available to agents

import random
import numpy as np
from scipy import spatial
from utils import distance2, unit_vector, vector_add, scalar_vector_product, vector_average


########################################################################################################################
#
#   PROGRAMS
#
########################################################################################################################

def random_reflex_generator(actions):
    def p_random_reflex(percept):
        if percept['Dirty']:
            return "GrabObject"
        elif percept['Bump']:
            return random.choice(['TurnRight', 'TurnLeft'])
        else:
            return random.choice(actions)
    return p_random_reflex


def _old_greedy_agent_generator():
    def p_old_greedy_agent(percepts):
        if percepts['Dirty']:
            return "Grab"
        else:
            dirts = [o[1] for o in percepts['Objects'] if o[0] == 'Dirt']
            agent_location = percepts['GPS']
            agent_heading = percepts['Compass']
            if dirts:
                nearest_dirt = find_nearest(agent_location, dirts)
                command = go_to(agent_location, agent_heading, nearest_dirt, percepts['Bump'])
                return command
            return ''
    return p_old_greedy_agent


def kmeans_roomba_generator():
    def p_kmeans_roomba(percepts):
        if percepts['Dirty']:
            return 'GrabObject'
        else:
            agent_location = (0, 0)
            agent_heading = percepts['Compass']
            # collect communication data
            # use a set comprehension to remove duplicates and convert back to a list

            dirts = {o[1] for o in percepts['Objects'] if o[0] == 'Dirt'}
            vacuums = {o[1] for o in percepts['Objects'] if o[0] == 'Agent'}
            unoccupied_dirts = list(dirts-vacuums)

            if dirts or unoccupied_dirts:
                nearest_dirt = find_nearest(agent_location, list(dirts)) # unoccupied_dirts)
                command = go_to(agent_location, agent_heading, nearest_dirt, percepts['Bump'])
                return command

            return random.choice(['TurnRight', 'TurnLeft', 'MoveForward', 'MoveForward', 'MoveForward', 'MoveForward'])
    return p_kmeans_roomba


def greedy_roomba_generator():
    def p_greedy_roomba(percepts):
        if percepts['Dirty']:
            return 'GrabObject'
        else:
            agent_location = (0, 0)
            agent_heading = percepts['Compass']
            # collect communication data
            # use a set comprehension to remove duplicates and convert back to a list
            dirts = {o[1] for o in percepts['Objects'] if o[0] == 'Dirt'}
            vacuums = {o[1] for o in percepts['Objects'] if o[0] == 'Agent'}   # TODO: This needs a better way to detect vacuums
            unoccupied_dirts = list(dirts-vacuums)

            if unoccupied_dirts:
                nearest_dirt = find_nearest(agent_location, unoccupied_dirts)
                command = go_to(agent_location, agent_heading, nearest_dirt, percepts['Bump'])
                return command

            return random.choice(['TurnRight', 'TurnLeft', 'MoveForward', 'MoveForward', 'MoveForward', 'MoveForward'])
    return p_greedy_roomba

def greedy_drone_generator(sensor_radius):
    def p_greedy_drone(percepts):
        agent_location = (0,0)
        agent_heading = percepts['Compass']
        # collect communication data
        dirts = [o[1] for o in percepts['Objects'] if o[0] == 'Dirt']
        drones = [d[1] for d in percepts['Objects'] if d[0] == 'GreedyDrone']

        if dirts:
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
    return p_greedy_drone

def rule_program_generator():
    def p_rule_program(percept):
        state = interpret_input(percept)
        rule = rule_match(state, rules)
        action = rule.action
        return action
    return p_rule_program
########################################################################################################################
#
#   STATE ESTIMATORS
#
########################################################################################################################

def basic_state_estimator_generator():
    def se_basic_state_estimator(percepts, comms):
        if not 'Objects' in percepts: percepts['Objects'] = [] # if percepts['Objects'] is empty, initialize it as an empty list
        for comm in comms.values():
            if 'Objects' in comm:
                for o in comm['Objects']:
                    if (o[1][0] + comm['GPS'][0] - percepts['GPS'][0], o[1][1] + comm['GPS'][1] - percepts['GPS'][1]) == (0,0) and o[0] == 'Dirt':
                        print(o, comm['GPS'], percepts['GPS'])
                percepts['Objects'] += [(o[0],  # o[1] is the location tuple, and o[1][0] is the x and o[1][1] is the y
                            (o[1][0] + comm['GPS'][0] - percepts['GPS'][0], o[1][1] + comm['GPS'][1] - percepts['GPS'][1]))
                            for o in comm['Objects']]

        # convert dirts to a set to remove duplicates and convert back to a list
        percepts['Objects'] = list(set(percepts['Objects']))  # note: does not preserve order

        return percepts
    return se_basic_state_estimator
########################################################################################################################
#
#   HELPERS
#
########################################################################################################################

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
