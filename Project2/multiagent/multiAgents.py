# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        #print str(scores)
        #raw_input("Press Enter to continue...")
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        tmp = [manhattanDistance(newPos,newGhostStates[i].getPosition()) if newScaredTimes[i] <= 2 else 99999
                               for i in range(len(newScaredTimes))]
        if len(tmp) == 0:
            closestUnscared = 0
        else:
            closestUnscared = min(tmp)

        foodL = newFood.asList()
        if len(foodL) == 0:
            closestFood = 0
        else:
            closestFood = min([manhattanDistance(newPos,food) for food in foodL])

        #print "Food = " + str(closestFood)
        #print "unscared = " + str(closestUnscared)
        if closestUnscared > 2:
            closestUnscaredTresh = 0
        else:
            closestUnscaredTresh = closestUnscared
        isSamePos = 0
        if currentGameState.getPacmanPosition() == newPos:
            isSamePos = 1
        score = -(100000000 * closestUnscaredTresh + isSamePos * 10000 + 100 * len(newFood.asList()) + closestFood)


        #print str(newPos) + " = " + str(score)
        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def minMax(self, state, depth, index = 0, zeroLevel = True):
        if index == state.getNumAgents():
            index = 0
            depth -= 1

        if depth == 0 or state.isLose() or state.isWin():
            ret = self.evaluationFunction(state)
        else:
            actions = state.getLegalActions(index)

            ret = None
            bestAction = None

            for action in actions:
                score = self.minMax(state.generateSuccessor(index, action), depth, index + 1, False)
                if index == 0:
                    if ret is None or score > ret:
                        ret = score
                        bestAction = action
                else:
                    if ret is None or score < ret:
                        ret = score
                        bestAction = action
        if zeroLevel and index == 0:
            return [bestAction, ret]
        else:
            return ret

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        ax = self.minMax(gameState, self.depth)
        return ax[0]
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def alphaBeta(self, state, depth, par, index=0, zeroLevel = True): #par[0] = alpha par[1] = beta
        if index == state.getNumAgents():
            index = 0
            depth -= 1

        if depth == 0 or state.isLose() or state.isWin():
            ret = self.evaluationFunction(state)
        else:
            actions = state.getLegalActions(index)

            ret = None
            bestAction = None
            me = par[:]

            for action in actions:
                score = self.alphaBeta(state.generateSuccessor(index, action), depth, me, index + 1, False)
                if index == 0:# maximizer
                    if ret is None or score > ret:
                        ret = score
                        bestAction = action

                    me[0] = max(me[0], ret)

                    if me[0] > me[1]:
                        break

                else: # minimizer
                    if ret is None or score < ret:
                        ret = score
                        bestAction = action

                    me[1] = min(me[1], ret)

                    if me[0] > me[1]:
                        break
        if index == 0 or index != 1:
            #print "Im maximizer"
            par[1] = min(par[1], ret)

        else:
            #print "Im minimizer"
            par[0] = max(par[0], ret)

        #if not (depth == 0 or state.isLose() or state.isWin()): print str(me)
        #print str(par)
        #print "b"
        if zeroLevel and index == 0:
            return [bestAction, ret]
        else:
            return ret

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        ax = self.alphaBeta(gameState, self.depth, [-99999999, 99999999])
        return ax[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectMax(self, state, depth, index, zeroLevel):
        if index == state.getNumAgents():
            index = 0
            depth -= 1

        if depth == 0 or state.isLose() or state.isWin():
            ret = self.evaluationFunction(state)*1.0/1
        else:
            actions = state.getLegalActions(index)

            ret = None
            cc = 0
            bestAction = None


            for action in actions:
                score = self.expectMax(state.generateSuccessor(index, action), depth, index + 1, False)
                if index == 0:
                    if ret is None or score > ret or (score == ret and bestAction == 'Stop'):
                        ret = score
                        bestAction = action
                else:
                    if ret is None:
                        ret = 0
                    ret += score
                    cc += 1
            if index != 0:
                ret = ret * 1.0 / cc

        if zeroLevel and index == 0:
            return [bestAction, ret]
        else:
            return ret

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return self.expectMax(gameState, self.depth,0,True)[0]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    if newPos in [newGhostStates[i].getPosition() if newScaredTimes[i] == 0 else (-1,-1)
                  for i in range(len(newScaredTimes))]:
        return -9999999999999999999


    tmp = [manhattanDistance(newPos,newGhostStates[i].getPosition()) if newScaredTimes[i] <= 2 else 99999
                               for i in range(len(newScaredTimes))]
    if len(tmp) == 0:
        closestUnscared = 0
    else:
        closestUnscared = min(tmp)

    foodL = newFood.asList() + currentGameState.getCapsules()

    if len(foodL) == 0:
        closestFood = -999999999999999999
    else:
        closestFood = findClosest(currentGameState)

    if closestUnscared > 2:
        closestUnscaredTresh = 0
    else:
        closestUnscaredTresh = closestUnscared


    score = -(1000000 * closestUnscaredTresh + 1000 * len(foodL) + closestFood)

    return score

# Abbreviation
better = betterEvaluationFunction



def findClosest(problem):
    """Search the node that has the lowest combined cost and heuristic first."""
    walls = problem.getWalls()
    food = problem.getFood()
    dsts = {}
    container = util.Queue()
    container.push(problem.getPacmanPosition())
    dsts[problem.getPacmanPosition()] = 0
    while not container.isEmpty():
        cur = container.pop()

        for a in range(cur[0] - 1, cur[0] + 2):
            for b in range(cur[1] - 1, cur[1] + 2):
               # print "-"+str((a,b))

                if (a, b) != cur and (a == cur[0] or b == cur[1]) and a >= 0 and b >= 0:
                    if a < walls.width and b < walls.height:

                        if not walls[a][b] and ((a,b) not in dsts):
                            dsts[(a,b)] = dsts[cur] + 1
                            if food[a][b] or (a,b) in problem.getCapsules():
                                return dsts[(a,b)]
                            container.push((a, b))

    return 0
