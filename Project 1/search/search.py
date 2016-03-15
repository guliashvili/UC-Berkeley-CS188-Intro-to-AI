# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import sys

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def get(map,last):
    ret = []
    
    while map[getStr(last)][1] != None and last != None:
        #print map[getStr(last)][1]
        ret.append(map[getStr(last)][1])
        last = map[getStr(last)][2]
    ret.reverse()
    return ret

def dfss(map,cur,problem):

    if problem.isGoalState(cur):
        return cur

    for elem in problem.getSuccessors(cur):
        if(getStr(elem[0]) not in map) :
            map[getStr(elem[0])] = [map[getStr(cur)][0] + 1,elem[1],cur]
            rt = dfss(map,elem[0],problem)
            if rt is not None:
                return rt

    return None

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    map = {}
    map[getStr(problem.getStartState())] = [0, None, None]
    last = dfss(map, problem.getStartState(), problem)
    
    return get(map,last)

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    container = util.Queue()
    map = {}
    container.push(problem.getStartState())
    map[getStr(problem.getStartState())] = [0,None,None]


    last = None
    while not container.isEmpty():
        cur = container.pop()
        if problem.isGoalState(cur):
            last = cur
            break
        for elem in problem.getSuccessors(cur):
            if(getStr(elem[0]) not in map) :
                map[getStr(elem[0])] = [map[getStr(cur)][0] + 1,elem[1],cur]
                container.push(elem[0])

    return get(map,last)
    
    
    return generalSearch(problem,queue);
# stackoverflow.com/questions/406121/flattening-a-shallow-list-in-python
def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result
def getStr(lst):
    return str([str(elem) for elem in flatten(lst)])

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    return aStarSearch(problem)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0
                  
                  
                  

""" #heuristic(elem[0],problem)"""
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    pqueue = util.PriorityQueue()

    map = {}
    gotMap = {}
    pqueue.push(problem.getStartState(),(0,0))
    map[getStr(problem.getStartState())] = (0,None,None)


    last = None
    while not pqueue.isEmpty():
        cur = pqueue.pop()
        if getStr(cur) in gotMap:
            continue

        gotMap[getStr(cur)] = 1

        if problem.isGoalState(cur):
            last = cur
            break


        for elem in problem.getSuccessors(cur):
            dst = map[getStr(cur)][0] + elem[2]
            if getStr(elem[0]) in gotMap:
                if map[getStr(elem[0])][0] > dst:
                    sys.exit("some msg")
            elif getStr(elem[0]) not in map or map[getStr(elem[0])][0] > dst:
                map[getStr(elem[0])] = (dst,elem[1],cur)
                pqueue.push(elem[0],(dst + heuristic(elem[0],problem), -dst))

    ret = []


    while map[getStr(last)][1] != None and last != None:
        ret.append(map[getStr(last)][1])
        last = map[getStr(last)][2]
    ret.reverse();
    return ret;



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
