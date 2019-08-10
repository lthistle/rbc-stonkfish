import os, sys, time, functools
from reconchess import *
import chess.engine
import numpy as np
import scipy.signal
from scipy.interpolate import interp1d

#TODO: sliding, removing boards from sliding information

### CONSTANTS ###

STOCKFISH_ENV_VAR = 'STOCKFISH_EXECUTABLE'
UNICODE_MAP = {chess.Piece(p, c).unicode_symbol():chess.Piece(p, c).symbol() for p in range(1, 7) for c in [True, False]}
PIECE_VALS = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 100}
FILTER = np.array([1]*9).reshape(3, 3)

MIN_TIME = 10
MAX_TIME = 30
MAX_MOVE_COUNT = 12000

# difference between winning on this turn and winning on the next turn
WIN = 10**5
MATE = WIN/2
LOSS_WIN_RATIO = 2
LOSE = -WIN * LOSS_WIN_RATIO
MATED = -MATE * LOSS_WIN_RATIO

# whether the opponent is able to pass or not
PASS = True

VERBOSE = 10
WIN_MSG = "Bot wins!"
LOSE_MSG = "Bot loses!"

### DECORATORS ###

def timer(f):
    """ Times the output of a function. """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        val = f(*args, **kwargs)
        print(f"Time taken: {1000*(time.time() - start)} (ms)")
        return val

    return wrapper

def silence(f):
    """ Ignores error messages. """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        sys.stderr = DevNull()
        val = f(*args, **kwargs)
        sys.stderr = sys.__stderr__
        return val

    return wrapper

def cache(f):
    """ Caches the output of a function. """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        k, param = f.__name__, args[1:] if isinstance(args[0], GhostBot) else args
        param = " ".join(map(str, param[:NARGS.get(k, len(param))]))

        if param not in CACHE[k]:
            CACHE[k][param] = f(*args, **kwargs)

        return CACHE[k][param]

    return wrapper

def make_board(board: chess.Board, move: chess.Move) -> chess.Board:
    """ Applies a move on a copied board. """
    temp = board.copy()
    if move is not None:
        temp.push(move)
    else:
        temp.turn = not temp.turn
    return temp

def set_turn(board: chess.Board, turn: bool) -> chess.Board:
    """ Sets the turn of a board. """
    board.turn = turn
    return board

def find_time(node_count: int) -> chess.engine.Limit:
    """ Gives the limitation on the engine per node. """
    time_func = interp1d([1, MAX_MOVE_COUNT], [MIN_TIME, MAX_TIME])
    # Equivalent to (MAX_TIME - MIN_TIME)/(MAX_MOVE_COUNT - 1)*(np.arange(1, MAX_MOVE_COUNT) - 1)[node_count] + MIN_TIME
    time_to_analyze = time_func(min(node_count, MAX_MOVE_COUNT))
    time_per_node = time_to_analyze/node_count
    print(f"{time_to_analyze:.3} seconds total, {time_per_node:.3} seconds per node")
    return chess.engine.Limit(time=time_per_node)

def time_str(t: float) -> str:
    """ Converts a time in seconds to a formatted string. """
    min = int(t/60)
    return f"{min} min {int(t - 60*min)} sec"

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

    @cache
    @silence
    def stkfsh_eval(self, board: chess.Board, move: chess.Move, limit: chess.engine.Limit) -> float:
        """ Evaluates a move via stockfish. """
        temp = make_board(board, move)

        # # probably because of turn stuff
        # sign = 1 if temp.is_valid() else -1
        # temp.turn = temp.turn if temp.is_valid() else not temp.turn

        info = self.engine.analyse(temp, limit)
        return info["score"].pov(self.color).score(mate_score=MATE)

    @cache
    def evaluate(self, board: chess.Board, move: chess.Move):
        """ Evaluates a move via RBC-specific rules. """
        points = None
        if move is not None and move.to_square == board.king(self.opponent_color):
            points = WIN
        elif board.is_checkmate():
            points = MATED
        elif board.is_check():
            # move that keeps the king in check, i.e. opponent can take king after this move
            if move not in board.legal_moves:
                points = LOSE

        return points

    def print_turn(self):
        """ Pretty print the turn. """
        msg = "Turn #{}: {}".format(self.turn_number, 'Black' if self.turn_number % 2 == 0 else 'White')
        print('*'*10 + msg + '*'*10)

    def remove_boards(self):
        """ If there are too many boards to check in a reasonable amount of time, check the 100 most 'at-risk' """
        if len(self.states) > 100:
            sort_list = []
            for x in range(len(self.states)):
                board_to_eval = self.states[x]
                #see if opponent's pieces are in position to attack
                board_to_eval.turn = self.opponent_color
                b_score = 0
                for m in board_to_eval.legal_moves:
                    dest = m.to_square
                    p = board_to_eval.piece_at(dest)
                    if p is not None:
                        b_score += PIECE_VALS[p.piece_type]
                sort_list.append((b_score, x))
                #revert back to proper player's turn
                board_to_eval.turn = self.color
            sort_list.sort()
            print("Analyzing 100 most at-risk boards")
            return [self.states[sort_list[x][1]] for x in range(100)]

        return self.states

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.color = color
        self.states = [board]
        self.opponent_color = not self.color
        self.turn_number = 1 #updated at beginning of handle_opp_move_result and at end of handle_move

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        if self.color and self.turn_number == 1:
            self.print_turn()
            return

        new_states = []
        states_set = set()
        for state in self.states:
            state.turn = self.opponent_color

        if captured_my_piece:
            for state in self.states:
                attackers = state.attackers(self.opponent_color, capture_square)
                for attacker_square in attackers:
                    temp = make_board(state, chess.Move(attacker_square, capture_square))
                    if str(temp) not in states_set:
                        states_set.add(str(temp))
                        new_states.append(temp)
        else:
            for state in self.states:
                #Assume these are all legal moves, still need to include castling
                for move in state.pseudo_legal_moves:
                    # Capture didn't happen this turn
                    if not state.is_capture(move):
                        temp = make_board(state, move)
                        if str(temp) not in states_set:
                            states_set.add(str(temp))
                            new_states.append(temp)

        self.states = new_states + ([set_turn(state, self.color) for state in self.states if str(state) not in states_set] if PASS else [])
        print("Number of states after handling opponent move: ", len(self.states))
        self.turn_number += 1
        self.print_turn()

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> Optional[Square]:
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

        return np.argmax(scipy.signal.convolve2d(expected.reshape(8, 8), FILTER)[1:-1, 1:-1]).item()

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

        print(f"Number of states after sensing: {len(self.states)}")
        if len(self.states) < VERBOSE:
            self.print_states()

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        table = {}
        start = time.time()

        # Ability to pass
        move_actions.append(None)

        node_count = len(move_actions)*len(self.states)
        print(f"Number of nodes to analyze: {node_count}")
        limit = find_time(node_count)

        # If only one board just let stockfish play it
        if len(self.states) == 1:
            board = self.states[0]
            # overwrite stockfish only if we are able to take the king this move
            for move in board.pseudo_legal_moves:
                assert board.turn == self.color
                if move.to_square == board.king(self.opponent_color):
                    return move
            result = self.engine.play(board, chess.engine.Limit(time=(MIN_TIME + MAX_TIME)/2))
            best, score = result.move, result.info.get("score", "unknown")
        else:
            states = self.remove_boards()
            for move in move_actions:
                for state in states:
                    state.turn = self.color
                    # move is invalid and equivalent to a pass
                    move = move if move in state.pseudo_legal_moves else None
                    points = self.evaluate(state, move)

                    if points is None:
                        points = self.stkfsh_eval(state, move, limit)

                    # assuming probability is constant, may change later
                    table[move] = table.get(move, 0) + points/len(states)

            if len(table) == 0: return

            best = max(table, key=lambda move: table[move])
            score = table[best]

        print(best, score)
        print(f"Time left before starting calculations for current move: {time_str(seconds_left)}")
        print(f"Time left now: {time_str(seconds_left - time.time() + start)}")
        return best

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        for state in self.states:
            state.turn = self.color

        if taken_move is not None:
            self.states = list(filter(lambda board: taken_move in board.pseudo_legal_moves, self.states))
            if captured_opponent_piece:
                self.states = list(filter(lambda board: board.color_at(capture_square) == self.opponent_color, self.states))
        if requested_move is not None and requested_move != taken_move:
            self.states = list(filter(lambda board: requested_move not in board.pseudo_legal_moves, self.states))            

        print("Number of states after moving: ", len(self.states))
        self.turn_number += 1
        self.print_turn()

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        print(WIN_MSG if winner_color == self.color else LOSE_MSG)
        self.engine.quit()

    def print_states(self, states=None):
        if not states:
            states = self.states
        for state in states:
            print(''.join([UNICODE_MAP.get(x,x) for x in state.unicode()]) + "\n")

CACHE = {f.__name__: {} for f in [GhostBot.stkfsh_eval, GhostBot.evaluate]}
NARGS = {f.__name__: v for f, v in [(GhostBot.stkfsh_eval, 2)]}
