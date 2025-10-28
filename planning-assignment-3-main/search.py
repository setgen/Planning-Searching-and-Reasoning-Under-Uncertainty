# I use LLM Copilot while completeing assignments, and therefore Copilot did helped with some parts of this assignment.
# Code blocks where Copilot assisted from Assignment 2 remain unchaged, and blocks where Copilot assisted for Assignment
# 3 are marked below.

import numpy as np
import queue
from collections import deque
from game import BoardState, GameSimulator, Rules
import math

class Problem:
    """
    This is an interface which GameStateProblem implements.
    You will be using GameStateProblem in your code. Please see
    GameStateProblem for details on the format of the inputs and
    outputs.
    """

    def __init__(self, initial_state, goal_state_set: set):
        self.initial_state = initial_state
        self.goal_state_set = goal_state_set

    def get_actions(self, state):
        """
        Returns a set of valid actions that can be taken from this state
        """
        pass

    def execute(self, state, action):
        """
        Transitions from the state to the next state that results from taking the action
        """
        pass

    def is_goal(self, state):
        """
        Checks if the state is a goal state in the set of goal states
        """
        return state in self.goal_state_set

class GameStateProblem(Problem):

    def __init__(self, initial_board_state, goal_board_state, player_idx):
        """
        player_idx is 0 or 1, depending on which player will be first to move from this initial state.

        Inputs for this constructor:
            - initial_board_state: an instance of BoardState
            - goal_board_state: an instance of BoardState
            - player_idx: an element from {0, 1}

        How Problem.initial_state and Problem.goal_state_set are represented:
            - initial_state: ((game board state tuple), player_idx ) <--- indicates state of board and who's turn it is to move
              ---specifically it is of the form: tuple( ( tuple(initial_board_state.state), player_idx ) )

            - goal_state_set: set([tuple((tuple(goal_board_state.state), 0)), tuple((tuple(goal_board_state.state), 1))])
              ---in otherwords, the goal_state_set allows the goal_board_state.state to be reached on either player 0 or player 1's
              turn.
        """
        super().__init__(tuple((tuple(initial_board_state.state), player_idx)), set([tuple((tuple(goal_board_state.state), 0)), tuple((tuple(goal_board_state.state), 1))]))
        self.sim = GameSimulator(None)
        self.search_alg_fnc = None
        self.set_search_alg()

    def set_search_alg(self, alg=""):
        """
        If you decide to implement several search algorithms, and you wish to switch between them,
        pass a string as a parameter to alg, and then set:
            self.search_alg_fnc = self.your_method
        to indicate which algorithm you'd like to run.
        """
        self.search_alg_fnc = self.bfs_shortest_plan

    def get_actions(self, state: tuple):
        """
        From the given state, provide the set possible actions that can be taken from the state

        Inputs: 
            state: (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn

        Outputs:
            returns a set of actions
        """
        s, p = state
        np_state = np.array(s)
        self.sim.game_state.state = np_state
        self.sim.game_state.decode_state = self.sim.game_state.make_state()

        return self.sim.generate_valid_actions(p)

    def execute(self, state: tuple, action: tuple):
        """
        From the given state, executes the given action

        The action is given with respect to the current player

        Inputs: 
            state: is a tuple (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn
            action: (relative_idx, position), where relative_idx is an index into the encoded_state
                with respect to the player_idx, and position is the encoded position where the indexed piece should move to.
        Outputs:
            the next state tuple that results from taking action in state
        """
        s, p = state
        k, v = action
        offset_idx = p * 6
        return tuple((tuple( s[i] if i != offset_idx + k else v for i in range(len(s))), (p + 1) % 2))

    def bfs_shortest_plan(self):
        """
        Breadth-first search algorithm.

        Outputs:
            A list of (state, action) pairs.
        """
        start = self.initial_state
        if self.is_goal(start):
            return [(start, None)]

        # LLM Copilot suggested parts of these lines of code
        q = deque([start])
        parent = {start: None}
        parent_action = {start: None}

        visited = {start}

        while q:
            s = q.popleft()
            actions = self.get_actions(s)
            for a in actions:
                s_next = self.execute(s, a)
                if s_next in visited:
                    continue
                visited.add(s_next)
                parent[s_next] = s
                parent_action[s_next] = a
                if self.is_goal(s_next):
                    # Reconstruct path
                    path = []
                    cur = s_next
                    path.append((cur, None))
                    while parent[cur] is not None:
                        prev = parent[cur]
                        act = parent_action[cur]
                        path.append((prev, act))
                        cur = prev
                    path.reverse()
                    return path
                q.append(s_next)
        # End of all suggested code in this block 
        
        return [(start, None)]

    # LLM Copilot suggested or assisted with parts of this code block (lines 157-263)
    def is_termination_state(self, state):
        enc_state, _ = state
        self.sim.game_state.state = np.array(enc_state)
        self.sim.game_state.decode_state = self.sim.game_state.make_state()
        return self.sim.game_state.is_termination_state()

    def terminal_value(self, state, max_player_idx):
        enc_state, _ = state
        w_row = int(enc_state[5]) // 7
        b_row = int(enc_state[11]) // 7
        white_wins = (w_row == 7)
        black_wins = (b_row == 0)
        if white_wins or black_wins:
            if max_player_idx == 0:
                return math.inf if white_wins else -math.inf
            else:
                return math.inf if black_wins else -math.inf
        return None

    def heuristic(self, state, max_player_idx):
        # A heursitic that takes into max_player's ball closeness to goal as well as
        # how many open blocks it can pass to

        # Evaluate ball closeness to goal
        enc_state, _ = state
        w_row = int(enc_state[5]) // 7
        b_row = int(enc_state[11]) // 7
        progress_white = w_row
        progress_black = 7 - b_row
        base = progress_white - progress_black

        # Evaluate number of passing blocks
        self.sim.game_state.state = np.array(enc_state)
        self.sim.game_state.decode_state = self.sim.game_state.make_state()
        moves = Rules.single_ball_actions(self.sim.game_state, max_player_idx)
        opp_moves = Rules.single_ball_actions(self.sim.game_state, 1 - max_player_idx)
        passes = (len(moves) - len(opp_moves)) * 0.1

        white_val = base + passes
        return white_val if max_player_idx == 0 else -white_val

    def order_actions(self, state, actions, player_idx, max_player_idx):
        # Try to make moves that advance ball first to help with alpha-beta pruning
        def key(a):
            rel, pos = a
            row = pos // 7
            if rel == 5:
                advance = row if player_idx == 0 else (7 - row)
                return (0, -advance, pos)
            return (1, rel, pos)
        return sorted(actions, key=key)

    def adversarial_search_method(self, start_state, max_depth=4, max_player_idx=0):
        transportation_table = {}  # The idea of using a transportation table was helped via a ChatGPT query about speeding up alpha-beta pruning

        def minimax(state, depth, alpha, beta, maximizing):
            terminal_value = self.terminal_value(state, max_player_idx)
            if terminal_value is not None:
                return None, terminal_value
            if depth == 0:
                return None, self.heuristic(state, max_player_idx)

            actions = self.get_actions(state)
            player_idx = state[1]
            actions = self.order_actions(state, actions, player_idx, max_player_idx)

            key = (state, depth, maximizing)
            if key in transportation_table:
                return transportation_table[key]

            if maximizing:
                best_value = -math.inf
                best_action = None
                for a in actions:
                    s_next = self.execute(state, a)
                    _, value = minimax(s_next, depth - 1, alpha, beta, not maximizing)
                    if value > best_value:
                        best_value, best_action = value, a
                    alpha = max(alpha, value)
                    if beta <= alpha:
                        break
                transportation_table[key] = (best_action, best_value)
                return best_action, best_value
            else:
                best_value = math.inf
                best_action = None
                for a in actions:
                    s_next = self.execute(state, a)
                    _, value = minimax(s_next, depth - 1, alpha, beta, not maximizing)
                    if value < best_value:
                        best_value, best_action = value, a
                    beta = min(beta, value)
                    if beta <= alpha:
                        break
                transportation_table[key] = (best_action, best_value)
                return best_action, best_value

        maximizing = (start_state[1] == max_player_idx)
        action, value = minimax(start_state, max_depth, -math.inf, math.inf, maximizing)

        if action is None:
            actions = list(self.get_actions(start_state))
            if actions:
                action = self.order_actions(start_state, actions, start_state[1], max_player_idx)[0]
            else:
                action = (0, start_state[0][0])
        return action, value
    # End of all suggested or assisted code in this block 
