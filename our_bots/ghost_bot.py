import os, sys, time, functools, logging
from reconchess import *
import reconchess.utilities as util
import chess.engine
import numpy as np
import scipy.signal
from scipy.interpolate import interp1d

### CONSTANTS ###

STOCKFISH_ENV_VAR = "STOCKFISH_EXECUTABLE"
UNICODE_MAP = {chess.Piece(p, c).unicode_symbol(): chess.Piece(p, c).symbol() for p in range(1, 7) for c in [True, False]}
PIECE_VALS = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 100}
FILTER = np.array([1]*9).reshape(3, 3)

MIN_TIME = 10
MAX_TIME = 30
MAX_NODE_COUNT = 12000
BOARD_LIMIT = 100

# difference between winning on this turn and winning on the next turn
WIN = 10**4
MATE = WIN/2
LOSS_WIN_RATIO = 10
LOSE = -WIN*LOSS_WIN_RATIO
MATED = -MATE*LOSS_WIN_RATIO

# whether the opponent is able to pass or not
PASS = True
# whether in replay enviroment, what color the replay is in (to save 50% of the time)
REPLAY, REPLAYCOLOR = False, chess.BLACK

logging.basicConfig(format="[%(asctime)s]%(levelname)s:%(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.CRITICAL)
logging.getLogger("urllib3.connectionpool").setLevel(logging.DEBUG)

fh = logging.FileHandler("log.log")
formatter = logging.Formatter("%(message)s")
fh.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setFormatter(formatter)

logger = logging.getLogger("StonkFish")
logger.addHandler(fh)
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)
logger.propagate = False

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

### CHESS UTILITY ###

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

def get_moves(board: chess.Board) -> List[chess.Move]:
    """ Accounts for the ability to castle through check. """
    return list(set(board.pseudo_legal_moves) | set(move for move in util.moves_without_opponent_pieces(board) if util.is_psuedo_legal_castle(board, move)))

def result(state: chess.Board, move: chess.Move) -> Optional[chess.Move]:
    """ Accounts for the "sliding" property unique to RBC chess. """
    if move in get_moves(state) or move is None:
        return move

    piece = state.piece_at(move.from_square)
    if piece is not None and piece.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
        return util.slide_move(state, move)

def flip_square(square: chess.Square) -> chess.Square:
    """ Flips a square. """
    return chess.square(7 - chess.square_file(square), 7 - chess.square_rank(square))

def flip_board(board: chess.Board) -> chess.Board:
    """ Flips white's and black's perspectives on the board. """
    temp = board.copy()
    new_map = {}
    piece_map = temp.piece_map()
    for square in piece_map:
        piece = piece_map[square]
        piece.color = not piece.color
        square = flip_square(square)
        new_map[square] = piece
    temp.set_piece_map(new_map)
    return temp

### ALG SPECIFIC ###

@cache
def evaluate(board: chess.Board, move: chess.Move, color: chess.Color) -> float:
    """ Evaluates a move via RBC-specific rules. """
    points = None
    if move is not None and move.to_square == board.king(not color):
        points = WIN
    elif board.is_checkmate():
        points = LOSE
    # move that keeps the king in check, i.e. opponent can take king after this move
    elif set_turn(make_board(board, move), board.turn).is_check():
        points = LOSE

    return points

def find_time(node_count: int) -> chess.engine.Limit:
    """ Gives the limitation on the engine per node. """
    time_func = interp1d([1, MAX_NODE_COUNT], [MIN_TIME, MAX_TIME])
    # Equivalent to (MAX_TIME - MIN_TIME)/(MAX_NODE_COUNT - 1)*(np.arange(1, MAX_NODE_COUNT) - 1)[node_count] + MIN_TIME
    time_to_analyze = time_func(min(node_count, MAX_NODE_COUNT)).item()
    time_per_node = time_to_analyze/node_count
    logger.info(f"{time_to_analyze:.3} seconds total, {time_per_node:.3} seconds per node")
    return chess.engine.Limit(time=time_per_node)

### FORMATTING ###

def print_turn(turn: int) -> None:
    """ Pretty print the turn. """
    msg = "Turn #{}: {}".format(turn, "Black" if turn % 2 == 0 else "White")
    logger.info("*"*10 + msg + "*"*10)

def time_str(t: float) -> str:
    """ Converts a time in seconds to a formatted string. """
    min = int(t/60)
    return f"{min} min {int(t - 60*min)} sec"

def print_states(boards: List[chess.Board]) -> None:
    if len(boards) < VERBOSE:
        for board in boards:
            logger.debug(board.board_fen())
            logger.debug("\n" + "".join([UNICODE_MAP.get(x, x) for x in board.unicode()]) + "\n")

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
    def stkfsh_eval(self, board: chess.Board, move: chess.Move, limit: chess.engine.Limit) -> float:
        """ Evaluates a move via stockfish. """
        temp = make_board(board, move)

        # # probably because of turn stuff
        # sign = 1 if temp.is_valid() else -1
        # temp.turn = temp.turn if temp.is_valid() else not temp.turn
        temp.clear_stack()
        try:
            info = self.engine.analyse(temp, limit)
            score = info["score"].pov(self.color)
        except (IndexError, ValueError):
            logger.error("Caught Stockfish error as " + str(self.color) + " (attempting to refresh then analyse again)")
            # Refresh engine and retry, should work. If not, default to 0.
            self.engine = chess.engine.SimpleEngine.popen_uci(os.environ[STOCKFISH_ENV_VAR])
            info = self.engine.analyse(temp, limit)
            if "score" not in info:
                logger.critical("Double failure, defaulting to 0.")
                return 0
            score = info["score"].pov(self.color)
        except Exception:
            self.engine = chess.engine.SimpleEngine.popen_uci(os.environ[STOCKFISH_ENV_VAR])
            return 0

        if score.is_mate():
            if score.score(mate_score=MATE) > 0:
                return score.score(mate_score=MATE)
            else:
                return score.score(mate_score=-MATED)
        return score.score()

    def remove_boards(self, use_stockfish: bool=False) -> List[chess.Board]:
        """ If there are too many boards to check in a reasonable amount of time, check the most 'at-risk'. """
        if len(self.states) > BOARD_LIMIT:
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
            logger.warning(f"Analyzing the {BOARD_LIMIT} most at-risk boards")

            if use_stockfish:
                return sorted(self.states, key=lambda board: self.stkfsh_eval(set_turn(board, self.opponent_color), None, chess.engine.Limit(depth=0)))
            return [self.states[sort_list[x][1]] for x in range(BOARD_LIMIT)]

        return self.states

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.color = color
        self.states = [board]
        self.opponent_color = not self.color
        #updated at beginning of handle_opp_move_result and at end of handle_move
        self.turn_number = 1
        logger.info(f"Playing against: {opponent_name}")

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        if self.color and self.turn_number == 1:
            print_turn(self.turn_number)
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
                #Assume these are all legal moves
                for move in get_moves(state):
                    # Capture didn't happen this turn
                    if not state.is_capture(move):
                        temp = make_board(state, move)
                        if str(temp) not in states_set:
                            states_set.add(str(temp))
                            new_states.append(temp)
            new_states += ([set_turn(state, self.color) for state in self.states if str(state) not in states_set] if PASS else [])

        self.states = new_states
        logger.info(f"Number of states after handling opponent move: {len(self.states)}")
        self.turn_number += 1
        print_turn(self.turn_number)

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

        logger.info(f"Number of states after sensing: {len(self.states)}")
        print_states(self.states)

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        # skip analyzing if in replay and colors don't match
        if REPLAY and self.turn_number % 2 != REPLAYCOLOR:
            return

        table = {}
        start = time.time()

        # Ability to pass
        moves = list(set(move for board in self.states for move in get_moves(board))) + [None]

        # Refresh engine
        self.engine = chess.engine.SimpleEngine.popen_uci(os.environ[STOCKFISH_ENV_VAR])
        # If only one board just let stockfish play it
        if len(self.states) == 1:
            board = self.states[0]
            # overwrite stockfish only if we are able to take the king this move
            # OR in checkmate but are able to castle out
            for move in get_moves(board):
                # assert board.turn == self.color
                if move.to_square == board.king(self.opponent_color):
                    return move
                elif board.is_checkmate() and util.is_psuedo_legal_castle(board, move):
                    return move

            # weird UCI exception stuff on valid board
            try:
                #Color flipping stuff if playing as black
                temp = board.copy()
                if self.color == chess.BLACK:
                    # assert flip_board(flip_board(temp)) == temp
                    temp = flip_board(board)
                    temp.turn = chess.WHITE
                temp.clear_stack()
                r = self.engine.play(temp, chess.engine.Limit(time=(MIN_TIME + MAX_TIME)/2))
                best, score = r.move, r.info.get("score", "unknown")
                if self.color == chess.BLACK:
                    best = chess.Move(flip_square(best.from_square), flip_square(best.to_square))
            # default to move analysis
            except Exception as e:
                logger.error("Caught Stockfish error as " + str(self.color) + " (move may not be accurate)")
                logger.error("Error: " + str(e))
                limit = find_time(len(moves))
                table = {move: (evaluate(board, move, self.color) if evaluate(board, move, self.color) is not None else self.stkfsh_eval(board, move, limit))
                         for move in moves}
                best = max(table, key=lambda move: table[move])
                score = table[best]
        else:
            states = self.remove_boards()
            node_count = len(moves)*len(states)
            logger.info(f"Number of nodes to analyze: {node_count}")
            limit = find_time(node_count)

            for move in moves:
                for state in states:
                    state.turn = self.color
                    # move is invalid and equivalent to a pass
                    new_move = result(state, move)
                    points = evaluate(state, new_move, self.color)

                    if points is None:
                        points = self.stkfsh_eval(state, new_move, limit)

                    # assuming probability is constant, may change later
                    table[move] = table.get(move, 0) + points/len(states)

            if len(table) == 0: return

            best = max(table, key=lambda move: table[move])
            score = table[best]

        logger.info(f"{best} {score if isinstance(score, str) else round(score, 2)}")
        logger.info(f"Time left before starting calculations for current move: {time_str(seconds_left)}")
        logger.info(f"Time left now: {time_str(seconds_left - time.time() + start)}")
        logger.debug(f"{ {str(k): round(v, 2) for k, v in table.items()} }")
        return best

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        for state in self.states:
            state.turn = self.color

        if requested_move is not None and requested_move != taken_move:
            #Requested move failed, so filter out any boards that allowed it to occur
            self.states = list(filter(lambda board: requested_move not in get_moves(board), self.states))
        if taken_move is not None:
            #Did some move, filter out any boards that didn't allow it to occur
            self.states = list(filter(lambda board: taken_move in get_moves(board), self.states))
            if captured_opponent_piece:
                #Captured something, filter out any boards that didn't allow it to occur
                self.states = list(filter(lambda board: board.color_at(capture_square) == self.opponent_color, self.states))

            for state in self.states:
                state.push(taken_move)

        logger.info(f"Number of states after moving: {len(self.states)}")
        self.turn_number += 1
        print_turn(self.turn_number)

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        logger.info(WIN_MSG if winner_color == self.color else LOSE_MSG)
        self.engine.quit()

CACHE = {f.__name__: {} for f in [GhostBot.stkfsh_eval, evaluate]}
NARGS = {f.__name__: v for f, v in [(GhostBot.stkfsh_eval, 2)]}
