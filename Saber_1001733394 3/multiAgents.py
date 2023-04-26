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
        # get the legal moves and the successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
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

        "*** YOUR CODE HERE ***"
        number_Capsules = len(currentGameState.getCapsules())
        minFood =float("inf")
        #Checks to see whether it is time to end the game
        if currentGameState.isLose():
            return -minFood
        elif currentGameState.isWin():
            return minFood
        elif action == Directions.STOP:
            return -minFood
        #get the nearest food
        for food in newFood.asList():
            minFood = min(minFood, manhattanDistance(newPos, food))

        #keep safe from the ghost if too close
        for ghost in successorGameState.getGhostPositions():
           man_ghost = (manhattanDistance(newPos, ghost))
           #print(man_ghost)
           #keep distance from chance to be  collided COLLISION_TOLERANCE = 0.7
           if (man_ghost < 3 and newScaredTimes == 0):
               return -minFood
        #return the new function to 
        return successorGameState.getScore() + 1.0/(minFood)+sum(newScaredTimes)+number_Capsules#make the function better with some constant value as evaliatio is w1n1+w2n2+....
       # return successorGameState.getScore()

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
        "*** YOUR CODE HERE ***"



   
    # MINIMAX ALGOIRTHM
    # function MINIMAX-DECISION(state) returns an action
    #return arg max_actions(s) min-value(result(state,a))
    
   # function max_value(state) returns a utility value
   # if terminal-tests(state) then return utility(state)
   # v = -infinity
   # for each a in actions(state) do 
   # v = max(v, min-value(results(s,a))) return v
    
    #function min-value(state) returns a utility value
    #if terminal-tests(state) then return utlity(state)
  #  v = infinity
   # for each a in action(state) do
   # v = min(v, max-value(result(s,a)))
  #  return v

  

        #a single search is consisted with a pac  move and the ghost's move
        #depth 2 search will be pacman and the ghost moving two 
        #function for maximizing
        def maxValue(state, depth):
            v = float("-inf")
            # Terminal condition to leave
            legalActions = state.getLegalActions(0)
            # NO legalActions means terminal state of game (win or lose)
            if not legalActions or depth == self.depth:
                return self.evaluationFunction(state)

            for action in legalActions: #ret val for each action
                v = max(v, minValue(state.generateSuccessor( 0, action), 0 + 1, depth + 1))
            return v



        def minValue(state, agentIndex, depth):
            v=float("inf")
            legalActions = state.getLegalActions(agentIndex)
             # NO legalActions means terminal state of game (win or lose)
            if not legalActions:
                return self.evaluationFunction(state)

            # When all ghosts moved, it's pacman's turn
            if agentIndex == state.getNumAgents() - 1:
                #Run for ghosts and get their minimum, if last ghost is reached and depth not getting yet , call max  
                for action in legalActions:
                    v = min(v,maxValue(state.generateSuccessor(agentIndex, action), depth ))
                return v
              
            else:
                for action in legalActions:
                     v = min(v,minValue(state.generateSuccessor(agentIndex, action),agentIndex+1, depth ))
                return v
               


      #initialize a variablevalue v to compare
         
      #get the new value (the max of the min)

        bestAction = 0
      #current value
        v = float("-inf")#initlize it to be negative infinity
      
     	#loop of legal actions to find the best action to take
        for action in gameState.getLegalActions(0):
      	  #initialize a variable to compare
          v1 = v
          #get the max of the min
          v = max(v,  minValue(gameState.generateSuccessor(0, action), 1, 1))

          #if the current value is better than before
          if v > v1:
          	  #update the best action
              bestAction = action

        return bestAction #return best action
      
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #similar to the minimax agent max min , except  alpha and beta values are used for pruning
        

        def minValue(state, agentIndex, depth, alpha, beta):
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                 # NO legalActions means terminal state of game (win or lose)
                return self.evaluationFunction(state)

            v = float('inf')
            if agentIndex == state.getNumAgents() - 1:
                for action in legalActions:
                    v = min(v,maxValue(state.generateSuccessor(agentIndex, action), depth,alpha,beta ))
                    if v < alpha:#compare
                      return v#pruning
                    
                    beta=min(beta,v)#update
            else:
                for action in legalActions:
                     v = min(v,minValue(state.generateSuccessor(agentIndex, action),agentIndex+1,depth,alpha,beta))
                     if v<alpha:
                       return v
                  
                     beta = min(beta, v)
            return v
               

        def maxValue(state, depth, alpha, beta):
            legalActions = state.getLegalActions(0)
            if not legalActions or depth == self.depth:
                 # NO legalActions means terminal state of game (win or lose)
                return self.evaluationFunction(state)

            v = float('-inf')
         
          
            for action in legalActions:
                v = max(v,  minValue(state.generateSuccessor(0, action),0+1, depth+1,alpha,beta))#get the max
              
                if v > beta:
               
                  return v
                alpha = max(alpha, v)

        
            return v
      
        v = float("-inf") # to start
        alpha = float("-inf")#alpha(neg inf)
        beta = float("inf")  # positive inf
        bestAction = 0  # best action is null
     #go though possible actions
        

        for action in gameState.getLegalActions(0):
           
           # to compare the values
           currv = minValue(gameState.generateSuccessor(0, action), 0+1, 1, alpha, beta)
           if currv > v:  # for getting the maximize value,update the value
             v = currv
             bestAction=action
           if currv>beta:
             return bestAction
            # and update the best action
         
           alpha = max(alpha, currv)  # update alpha
        return bestAction
      

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def maxValue(state, depth):
            legalActions = state.getLegalActions(0)

            # NO legalActions means terminal state of game (win or lose)
            if not legalActions or depth == self.depth:
              return self.evaluationFunction(state)
            v=float("-inf")
            #get max action utitily
            for action in legalActions:
              v = max(v,expectiValue(state.generateSuccessor(0, action), 0 + 1, depth + 1) )
            return v

        def expectiValue(state, agentIndex, depth):
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            v = 0
            
            if agentIndex == state.getNumAgents() - 1:
                # it differs from the minValue used in Minimax.
                #pac has to has calculate the expectvalue using the probabilities  to explore 
                for action in legalActions:
                    v += maxValue(state.generateSuccessor(agentIndex,action), depth) 
            else:
                for action in legalActions:
                    v += expectiValue(state.generateSuccessor(agentIndex,action), agentIndex + 1, depth) 
            return v*(1.0 / len(legalActions))#probability
         
        
        v = float("-inf")  # initlize it to be negative infinity
        bestAction=0
	#go through all legal actions so we can find the best action to take
        for action in gameState.getLegalActions():
      	  #initialize a variable to be equal to the current value v. this will allow us to compare
          v1=expectiValue(gameState.generateSuccessor(0, action), 1, 1)
          #get the new value (the max of the min)
        

          #if the current value is better than the previous
          if v1 > v:
              v=v1
        	  #update the best action
              bestAction = action

        return bestAction  # return best action
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    #get the capsulelist to make func better
    capsuleList = currentGameState.getCapsules()
    #initialize lists
    distancesToFood=[]
    minghost=[]
 
  
    #  in 1 move the score gets decreased by 1
    value = currentGameState.getScore()

    # distance to ghosts
    for ghost in currentGameState.getGhostPositions():
      man_ghost = (manhattanDistance(newPos, ghost))
      minghost.append(man_ghost)
    if len(minghost):
            value -= min(minghost)


    # distance to closest food
    for food in newFood.asList():
        foodDist= manhattanDistance(newPos, food)
        distancesToFood.append(foodDist)
    if len(distancesToFood):
        value += 0.5/ min(distancesToFood)
#updae value(distofghost,distoffood,capsule)
    return value-(50*len(capsuleList))

# Abbreviation
better = betterEvaluationFunction

