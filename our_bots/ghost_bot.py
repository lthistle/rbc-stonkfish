import os, sys, itertools, time
from reconchess import *
import chess.engine
import numpy as np
import scipy.signal

STOCKFISH_ENV_VAR = 'STOCKFISH_EXECUTABLE'
UNICODE_MAP = {chess.Piece(p, c).unicode_symbol(): chess.Piece(p, c).symbol() for p in range(1, 7) for c in [True, False]}

FILTER = np.array([1]*9).reshape(3, 3)

SEC = 8
TIME = 5*10**-2
# difference between winning on this turn and winning on the next turn
WIN = 10**5
MATE = WIN/2

VERBOSE = 10

class DevNull:
    def write(self, msg):
        pass

class GhostBot(Player):
    def __init__(self):
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
        self.color = color
        self.states = [board]
        self.opponent_color = not self.color

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        new_states = []
        states_set = set()

        for state in self.states:
            state.turn = self.opponent_color

        if captured_my_piece:
            for state in self.states:
                attackers = state.attackers(self.opponent_color, capture_square)
                for attacker_square in attackers:
                    temp = state.copy()
                    move = chess.Move(attacker_square, capture_square)
                    temp.push(move)
                    if str(temp) not in states_set:
                        states_set.add(str(temp))
                        new_states.append(temp)
        else:
            for state in self.states:
                #Assume these are all legal moves, still need to include castling
                for move in state.pseudo_legal_moves:
                    # Capture didn't happen this turn
                    if not state.is_capture(move):
                        temp = state.copy()
                        temp.push(move)
                        if str(temp) not in states_set:
                            states_set.add(str(temp))
                            new_states.append(temp)

        self.states = new_states
        print("Number of states after handling opponent move: ", len(self.states))

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[Square]:
        expected = np.zeros((64,))
        for square in range(64):
            table = {}
            for state in self.states:
                piece = state.piece_at(square)
                if piece is None or piece.color == self.opponent_color:
                    typ = piece.piece_type if piece is not None else None
                    table[typ] = table.get(typ, 0) + 1
            for piece in table:
                expected[square] += table[piece]*(len(self.states) - table[piece])/len(self.states)

        return np.argmax(scipy.signal.convolve2d(expected.reshape(8, 8), FILTER)[1:-1, 1:-1])

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        confirmed_states = []
        for state in self.states:
            match = True
            for square, piece in sense_result:
                typ = piece.piece_type if piece else None
                if state.piece_type_at(square) != typ:
                    match = False
                    break
            if match:
                confirmed_states.append(state)
        self.states = confirmed_states

        print("Number of states after sensing: ", len(self.states))
        if len(self.states) < VERBOSE:
            self.print_states()

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        table = {}
        move_count = 0
        start = time.time()

        for state in self.states:
            state.turn = self.color
            for move in state.pseudo_legal_moves:
                move_count += 1

        time_to_analyze = max(SEC/move_count, TIME)
        for state in self.states:
            for move in state.pseudo_legal_moves:
                sys.stderr = DevNull()
                points = 0

                if move.to_square == state.king(self.opponent_color):
                    points = WIN
                else:
                    temp = state.copy()
                    temp.push(move)
                    temp.turn = self.opponent_color
                    info = self.engine.analyse(temp, chess.engine.Limit(time=time_to_analyze))
                    points = info["score"].pov(self.color).score(mate_score=MATE)
                if info["score"]:
                    # assuming probability is constant, may change later
                    table[move] = table.get(move, 0) + points/len(self.states)

                sys.stderr = sys.__stderr__

        if len(table) == 0: return

        best = max(table, key=lambda move: table[move])
        print(best, table[best])
        print("Time left: ", seconds_left - time.time() + start)
        return best

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        for state in self.states:
            state.turn = self.color
        if taken_move is not None:
            #Request passed, so remove all the boards that didn't allow that move to occur
            self.states = list(filter(lambda board: taken_move in board.pseudo_legal_moves, self.states))
            #If capture occured, remove all boards that didn't have an opponent piece there
            if captured_opponent_piece:
                self.states = list(filter(lambda board: board.color_at(capture_square) == self.opponent_color, self.states))
            for state in self.states:
                state.push(taken_move)
        elif requested_move is not None:
            #Request failed, so remove all the boards that allowed that move to occur
            self.states = list(filter(lambda board: requested_move not in board.pseudo_legal_moves, self.states))
        print("Number of states after moving: ", len(self.states))

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        if winner_color == self.color:
            print("BET WE WON BOYS")
        else:
            print("SHAFT WE LOST REEE")
        self.engine.quit()

    def print_states(self, states=None):
        if not states:
            states = self.states
        for state in states:
            print("".join([UNICODE_MAP.get(x, x) for x in state.unicode()]) + "\n")
