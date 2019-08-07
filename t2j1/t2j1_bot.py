import chess.engine
import random
from reconchess import *
import os
import yaml

home_path = os.path.join(os.path.dirname(__file__), '..')
stockfish_path = yaml.load(open(os.path.join(home_path, 'config.yaml')))['stockfish_path']
os.environ['STOCKFISH_EXECUTABLE'] = os.path.join(home_path, stockfish_path)
STOCKFISH_ENV_VAR = 'STOCKFISH_EXECUTABLE'


class T2J1Bot(Player):

    def __init__(self):
        self.board = None
        self.color = None
        self.my_piece_captured_square = None

        #check for stockfish environment variable
        if STOCKFISH_ENV_VAR not in os.environ:
            raise KeyError('T2J1Bot requires an environment variable called "{}" pointing to the Stockfish executable'.format(STOCKFISH_ENV_VAR))

        #check for executable
        stockfish_path = os.environ[STOCKFISH_ENV_VAR]
        if not os.path.exists(stockfish_path):
            raise ValueError('No stockfish executable found at "{}"'.format(stockfish_path))

        # initialize the stockfish engine
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.board = board
        self.color = color

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        pass

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> Optional[Square]:
        return None

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        pass

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        return None

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move], captured_opponent_piece: bool, capture_square: Optional[Square]):
        pass

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason], game_history: GameHistory):
        self.engine.quit()
