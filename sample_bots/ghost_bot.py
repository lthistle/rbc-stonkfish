import random
from reconchess import *
import chess.engine
import os
import numpy as np
import scipy

STOCKFISH_ENV_VAR = 'STOCKFISH_EXECUTABLE'

class GhostBot(Player):
    
    def __init__(self):
        self.board = None
        self.color = None
        self.states = []
        # make sure stockfish environment variable exists
        if STOCKFISH_ENV_VAR not in os.environ:
            raise KeyError(
                'GhostBot requires an environment variable called "{}" pointing to the Stockfish executable'.format(
                    STOCKFISH_ENV_VAR))

        # make sure there is actually a file
        stockfish_path = os.environ[STOCKFISH_ENV_VAR]
        if not os.path.exists(stockfish_path):
            raise ValueError('No stockfish executable found at "{}"'.format(stockfish_path))

        # initialize the stockfish engine
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.board = board
        self.color = color
        self.states = []
        self.opponent_color = not self.color

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        poss_moves = self.board.pseudo_legal_moves()
        new_states = []
        if self.captured_my_piece:
            self.board.remove_piece_at(capture_square)
            for state in self.states:
                self.state.turn = self.opponent_color
                attackers = state.attackers(color, capture_square)
                for attacker in attackers:
                    temp = state.copy()
                    temp.set_piece_at(capture_square, attacker)        
                    new_states.append(temp)                
        else:
            for state in self.states:
                self.state.turn = self.opponent_color
                for move in state.pseudo_legal_moves: #Assume these are all legal moves, still need to include castling
                    new_state = state.copy()
                    new_state.push(move)
                    if new_state.is_capture(move):
                        continue
                    new_states.append(new_state)
        self.states = new_states        

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[Square]:
        expected = np.zeros((64,))
        for square in expected:
            table = {}
            for state in self.states:
                piece = state.piece_at(square)
                if piece.color == self.color:
                    continue
                table[piece.piece_type] = 1 if piece.piece_type in table else table[piece.piece_type] + 1

            for piece in table:
                p = table[piece]/len(self.states)
                expected[square] += p*(len(self.states) - table[piece])

        scan = np.argmax(scipy.signal.convolve2d(np.pad(expected.reshape(8, 8), ((1,)*2,)*2, 'constant'), np.array([1]*9).reshape(3, 3)))
        return scan

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        confirmed_states = []
        for state in self.states:
            worked = True
            for square, piece in sense_result:
                if state.piece_type_at(square) != piece.piece_type:
                    worked = False
                    break
            if worked:
                confirmed_states.append(state)
        self.states = confirmed_states

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:

        return random.choice(move_actions + [None])

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        if taken_move is not None:
            #Request passed, so remove all the boards that didn't allow that move to occur
            self.states = self.filter_states(lambda board: taken_move in board.pseudo_legal_moves)
            #If capture occured, remove all boards that didn't have an opponent piece there
            if captured_opponent_piece:
                self.states = self.filter_states(lambda board: board.color_at(capture_square) == self.opponent_color)
            for state in self.states:
                state.push(taken_move)
        elif requested_move is not None:
            #Request failed, so remove all the boards that allowed that move to occur
            self.states = self.filter_states(lambda board: requested_move not in board.pseudo_legal_moves)

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        print("BET WE WON BOYS")
    
    def filter_states(self, func):
        filtered = []
        for state in self.states:
            if func(state):
                filtered.append(state)
        return filtered

bot = GhostBot()