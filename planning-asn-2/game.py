# LLM Copilot did helped with some parts of this assignment see below 

import numpy as np

class BoardState:
    """
    Represents a state in the game
    """

    def __init__(self):
        """
        Initializes a fresh game state
        """
        self.N_ROWS = 8
        self.N_COLS = 7

        self.state = np.array([1,2,3,4,5,3,50,51,52,53,54,52])
        self.decode_state = [self.decode_single_pos(d) for d in self.state]

    def update(self, idx, val):
        """
        Updates both the encoded and decoded states
        """
        self.state[idx] = val
        self.decode_state[idx] = self.decode_single_pos(self.state[idx])

    def make_state(self):
        """
        Creates a new decoded state list from the existing state array
        """
        return [self.decode_single_pos(d) for d in self.state]

    def encode_single_pos(self, cr: tuple):
        """
        Encodes a single coordinate (col, row) -> Z

        Input: a tuple (col, row)
        Output: an integer in the interval [0, 55] inclusive
        """
        col, row = cr
        return int(row) * self.N_COLS + int(col)

    def decode_single_pos(self, n: int):
        """
        Decodes a single integer into a coordinate on the board: Z -> (col, row)

        Input: an integer in the interval [0, 55] inclusive
        Output: a tuple (col, row)
        """
        n = int(n)
        col = n % self.N_COLS
        row = n // self.N_COLS

        return (col, row)

    def is_termination_state(self):
        """
        Checks if the current state is a termination state. Termination occurs when
        one of the player's move their ball to the opposite side of the board.

        You can assume that `self.state` contains the current state of the board, so
        check whether self.state represents a termainal board state, and return True or False.
        """
        if not self.is_valid():
          return False
        white_ball = int(self.state[5])
        black_ball = int(self.state[11])
        _, w_row = self.decode_single_pos(white_ball)
        _, b_row = self.decode_single_pos(black_ball)

        return (w_row == self.N_ROWS - 1) or (b_row == 0)

    def is_valid(self):
        """
        Checks if a board configuration is valid. This function checks whether the current
        value self.state represents a valid board configuration or not. This encodes and checks
        the various constrainsts that must always be satisfied in any valid board state during a game.

        If we give you a self.state array of 12 arbitrary integers, this function should indicate whether
        it represents a valid board configuration.

        Output: return True (if valid) or False (if not valid)        
        """
        max_cell_num = self.N_ROWS * self.N_COLS

        # Check if cell values are on grid
        for v in self.state:
            if v < 0 or v >= max_cell_num:
                return False

        #Set up blocks and balls
        white_blocks = [int(x) for x in self.state[0:5]]
        white_ball = int(self.state[5])
        black_blocks = [int(x) for x in self.state[6:11]]
        black_ball = int(self.state[11])
        
        # Check correct num of blocks
        if len(set(white_blocks)) != 5:
            return False
        if len(set(black_blocks)) != 5:
            return False

        # Ensure different locations
        if set(white_blocks) & set(black_blocks):
            return False

        # Check ball locations
        if white_ball not in set(white_blocks):
            return False
        if black_ball not in set(black_blocks):
            return False

        return True

class Rules:

    @staticmethod
    def single_piece_actions(board_state, piece_idx):
        """
        Returns the set of possible actions for the given piece, assumed to be a valid piece located
        at piece_idx in the board_state.state.

        Inputs:
            - board_state, assumed to be a BoardState
            - piece_idx, assumed to be an index into board_state, identfying which piece we wish to
              enumerate the actions for.

        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that piece_idx can move to during this turn.
        """
        N_ROWS, N_COLS = board_state.N_ROWS, board_state.N_COLS
        player_idx = 0 if piece_idx < 6 else 1
        ball_idx = 5 if player_idx == 0 else 11

        # Get the block's current pos
        pos = int(board_state.state[piece_idx])
        col, row = board_state.decode_single_pos(pos)

        # If the block has the ball, it can't move
        if int(board_state.state[ball_idx]) == pos:
            return set()

        # Get all the cells occupied by blocks
        occupied_cells = set(int(x) for x in list(board_state.state[0:5]) + list(board_state.state[6:11]))

        # All possible block moves
        all_moves = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]

        moves = set()
        for c, r in all_moves:
            new_col, new_row = col + c, row + r
            if 0 <= new_col < N_COLS and 0 <= new_row < N_ROWS:
                enc = board_state.encode_single_pos((new_col, new_row))
                if enc not in occupied_cells:
                    moves.add(enc)

        return moves

    @staticmethod
    def single_ball_actions(board_state, player_idx):
        """
        Returns the set of possible actions for moving the specified ball, assumed to be the
        valid ball for plater_idx  in the board_state

        Inputs:
            - board_state, assumed to be a BoardState
            - player_idx, either 0 or 1, to indicate which player's ball we are enumerating over
        
        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that player_idx's ball can move to during this turn.        
        """
        assert player_idx in (0,1)
        N_ROWS, N_COLS = board_state.N_ROWS, board_state.N_COLS
        ball_idx = 5 if player_idx == 0 else 11
        offset = 0 if player_idx == 0 else 6

        # Get set of friendly blocks
        friendly = [int(x) for x in board_state.state[offset:offset+5]]
        friendly_set = set(friendly)
        start = int(board_state.state[ball_idx])

        # Get set of all blocks that occupy squares
        occupied_blocks = set(int(x) for x in list(board_state.state[0:5]) + list(board_state.state[6:11]))

        # LLM Copilot suggested parts of these lines of code

        # Determine if the ball has a clear line of sight to target
        def clear_los(a_enc, b_enc):
            if a_enc == b_enc:
                return False

            a_col, a_row = board_state.decode_single_pos(a_enc)
            b_col, b_row = board_state.decode_single_pos(b_enc)
            diff_col = b_col - a_col
            diff_row = b_row - a_row

            # Check if same row, same column, or on the diagonal
            if not (a_row == b_row or a_col == b_col or abs(diff_col) == abs(diff_row)):
                return False

            step_col = 0 if diff_col == 0 else (1 if diff_col > 0 else -1)
            step_row = 0 if diff_row == 0 else (1 if diff_row > 0 else -1)
            c, r = a_col + step_col, a_row + step_row
            while (c, r) != (b_col, b_row):
                enc = board_state.encode_single_pos((c, r))
                if enc in occupied_blocks:
                    return False
                c += step_col
                r += step_row
            return True

        # Friendly blocks with clear line of sight
        adj = {u: set() for u in friendly}
        for i in range(len(friendly)):
            for j in range(i+1, len(friendly)):
                u, v = friendly[i], friendly[j]
                if clear_los(u, v):
                    adj[u].add(v)
                    adj[v].add(u)

        # Look through friendly blocks and determine which are valid moves
        reachable = set()
        if start in adj:
            q = [start]
            seen = {start}
            while q:
                u = q.pop(0)
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        q.append(v)
                        reachable.add(v)
        # Exclude the starting square
        if start in reachable:
            reachable.remove(start)

        # End of all suggested code in this block 

        return reachable

class GameSimulator:
    """
    Responsible for handling the game simulation
    """

    def __init__(self, players):
        self.game_state = BoardState()
        self.current_round = -1 ## The game starts on round 0; white's move on EVEN rounds; black's move on ODD rounds
        self.players = players

    def run(self):
        """
        Runs a game simulation
        """
        while not self.game_state.is_termination_state():
            ## Determine the round number, and the player who needs to move
            self.current_round += 1
            player_idx = self.current_round % 2
            ## For the player who needs to move, provide them with the current game state
            ## and then ask them to choose an action according to their policy
            action, value = self.players[player_idx].policy( self.game_state.make_state() )
            print(f"Round: {self.current_round} Player: {player_idx} State: {tuple(self.game_state.state)} Action: {action} Value: {value}")

            if not self.validate_action(action, player_idx):
                ## If an invalid action is provided, then the other player will be declared the winner
                if player_idx == 0:
                    return self.current_round, "BLACK", "White provided an invalid action"
                else:
                    return self.current_round, "WHITE", "Black probided an invalid action"

            ## Updates the game state
            self.update(action, player_idx)

        ## Player who moved last is the winner
        if player_idx == 0:
            return self.current_round, "WHITE", "No issues"
        else:
            return self.current_round, "BLACK", "No issues"

    def generate_valid_actions(self, player_idx: int):
        """
        Given a valid state, and a player's turn, generate the set of possible actions that player can take

        player_idx is either 0 or 1

        Input:
            - player_idx, which indicates the player that is moving this turn. This will help index into the
              current BoardState which is self.game_state
        Outputs:
            - a set of tuples (relative_idx, encoded position), each of which encodes an action. The set should include
              all possible actions that the player can take during this turn. relative_idx must be an
              integer on the interval [0, 5] inclusive. Given relative_idx and player_idx, the index for any
              piece in the boardstate can be obtained, so relative_idx is the index relative to current player's
              pieces. Pieces with relative index 0,1,2,3,4 are block pieces that like knights in chess, and
              relative index 5 is the player's ball piece.            
        """
        #if player_idx not in (0,1):
        #    raise ValueError("player_idx must be 0 or 1")
        if not self.game_state.is_valid():
            return set()

        actions = set()
        offset = 0 if player_idx == 0 else 6

        # Block moves
        for rel in range(5):
            piece_idx = offset + rel
            for pos in Rules.single_piece_actions(self.game_state, piece_idx):
                actions.add((rel, int(pos)))

        # Ball moves
        for pos in Rules.single_ball_actions(self.game_state, player_idx):
            actions.add((5, int(pos)))

        return actions

    def validate_action(self, action: tuple, player_idx: int):
        """
        Checks whether or not the specified action can be taken from this state by the specified player

        Inputs:
            - action is a tuple (relative_idx, encoded position)
            - player_idx is an integer 0 or 1 representing the player that is moving this turn
            - self.game_state represents the current BoardState

        Output:
            - if the action is valid, return True
            - if the action is not valid, raise ValueError        
        """
        # Type and structure checks
        if not isinstance(action, tuple) or len(action) != 2:
            raise ValueError("Action must be a tuple (relative_idx, encoded_position)")
        rel, pos = action
        if not isinstance(rel, (int, np.integer)):
            raise ValueError("relative_idx must be an integer in [0,5]")
        if rel < 0 or rel > 5:
            raise ValueError("relative_idx out of range [0,5]")
        if not isinstance(pos, (int, np.integer)):
            raise ValueError("encoded_position must be an integer")
        if not (0 <= int(pos) < self.game_state.N_ROWS * self.game_state.N_COLS):
            raise ValueError("encoded_position out of bounds")
        if player_idx not in (0,1):
            raise ValueError("player_idx must be 0 or 1")
        if not self.game_state.is_valid():
            raise ValueError("Current board configuration is invalid")

        offset = 0 if player_idx == 0 else 6

        if rel == 5:
            legal_targets = Rules.single_ball_actions(self.game_state, player_idx)
            if int(pos) not in legal_targets:
                raise ValueError("Ball move is not legal from current holder")
            return True
        else:
            piece_idx = offset + rel
            ball_idx = offset + 5
            if int(self.game_state.state[piece_idx]) == int(self.game_state.state[ball_idx]):
                raise ValueError("Block cannot move while holding the ball")
            legal_targets = Rules.single_piece_actions(self.game_state, piece_idx)
            if int(pos) not in legal_targets:
                raise ValueError("Block move is not a legal move to an empty square")
            return True
    
    def update(self, action: tuple, player_idx: int):
        """
        Uses a validated action and updates the game board state
        """
        offset_idx = player_idx * 6 ## Either 0 or 6
        idx, pos = action
        self.game_state.update(offset_idx + idx, pos)
