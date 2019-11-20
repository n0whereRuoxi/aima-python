# define the list of programs available to agents

import random, math
import numpy as np
from scipy import spatial
from utils import distance, distance2, unit_vector, vector_add, scalar_vector_product, vector_average


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
            agent_location = percepts['GPS']
            agent_heading = percepts['Compass']
            # collect communication data
            # use a set comprehension to remove duplicates and convert back to a list

            dirts = list({o[1] for o in percepts['Objects'] if o[0] == 'Dirt'})

            if dirts:
                vacuums = list({o[1] for o in percepts['Objects'] if o[0] == 'Agent'})
                (dirt_clusters, dirt_means) = k_means(dirts, len(vacuums), 10, list(vacuums))

                assignments = optimal_assignments(list(vacuums), dirt_means, dirt_clusters)

                my_cluster = dirt_clusters[dirt_means.index(assignments[vacuums.index(agent_location)])]
                if my_cluster == []:
                    nearest_dirt = assignments[vacuums.index(agent_location)]
                else:
                    nearest_dirt = find_nearest(agent_location, my_cluster)
                command = go_to(agent_location, agent_heading, nearest_dirt, percepts['Bump'])
                return command

            return random.choice(['TurnRight', 'TurnLeft', 'MoveForward', 'MoveForward', 'MoveForward', 'MoveForward'])
    return p_kmeans_roomba


def greedy_roomba_generator():
    def p_greedy_roomba(percepts):
        if percepts['Dirty']:
            return 'GrabObject'
        else:
            agent_location = percepts['GPS']
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
                #if ('Dirt', (9, 15)) in comm['Objects']: print('wtf =', comm['GPS'])
                percepts['Objects'] += comm['Objects']  # o[1] is the location tuple, and o[1][0] is the x and o[1][1] is the y
                # percepts['Objects'] += [(o[0],  # o[1] is the location tuple, and o[1][0] is the x and o[1][1] is the y
                #            (o[1][0] + comm['GPS'][0] - percepts['GPS'][0], o[1][1] + comm['GPS'][1] - percepts['GPS'][1]))
                #            for o in comm['Objects']]

        # convert dirts to a set to remove duplicates and convert back to a list
        #percepts['Objects'] = list(set(percepts['Objects']))  # note: does not preserve order

        return percepts
    return se_basic_state_estimator
########################################################################################################################
#
#   HELPERS
#
########################################################################################################################

def find_nearest(agent_location, dirts):
    dists = [distance2(agent_location, d) for d in dirts]
    return dirts[dists.index(min(dists))]

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


def k_means(X, k, n, init=None):
    # INITIALIZATION
    # return two vectors for the minimum and maximum point values for each dimension in the data

    # initialize the means by selecting random points from the dirts
    if init is None or len(init)!=k:
        if k <= len(X):
            means = random.sample(X,k)
        else:
            means = X + random.choices(X,k=k-len(X))
    else:
        if len(init)!=k: print('init is the incorrect size')
        means = init

    for i in range(n):
        dists = [[distance2(u,x) for u in means] for x in X]    # using distance squared because square roots
                                                                # are expensive and argmin(x) == argmin(x^2)
        clusters = [[x for (x, ds) in zip(X, dists) if ds.index(min(ds)) == i] for i in range(k)]

        means = [vector_average(xs) if xs else u for (xs,u) in zip(clusters, means)]

    return (clusters, means)


def optimal_assignments(ag_locs, means, clusters):
    # initialize the assignments
    assignments = random.sample(means,k=len(means))

    for iter in range(10):
        for i in range(len(means)):
            dists = []
            for j in range(len(means)):
                new_assignments = assignments.copy()
                new_assignments[j],new_assignments[i] = new_assignments[i],new_assignments[j]
                dists.append(cum_distance_of_assignment(ag_locs,new_assignments))
            best_j = dists.index(min(dists))
            assignments[best_j],assignments[i] = assignments[i],assignments[best_j]
    return assignments


def cum_distance_of_assignment(ag_locs, assignment):
    return sum([distance(loc,asgn) for (loc,asgn) in zip(ag_locs,assignment)])


if __name__ == '__main__':
    X = [(random.gauss(5,1), random.gauss(10,3)) for x in range(10)] + [(random.gauss(10,2), random.gauss(5,2)) for x in range(20)]
    print(X)
    (c,m) = k_means(X, 5, 5)

    print(c)
    print(m)

    agent_locs = [(5,5),(5,10),(10,5),(10,10),(7.5,7.5)]
    assignments = optimal_assignments(agent_locs,m,c)

    print('assignments =', list(zip(agent_locs, assignments)))

    import matplotlib.pyplot as plot

    plot.plot(*zip(*X),'b.')
    plot.plot(*zip(*m),'r*')
    plot.show()
