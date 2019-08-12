import random
from reconchess import *
import typing

class DefensiveBot(Player):

    def __init__(self):
        self.board = None
        self.color = None
        self.pwnloc = 7

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.board = board
        self.color = color

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        pass

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[Square]:
        return random.choice(sense_actions)

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        pass

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        self.pwnloc += 8
        if self.pwnloc == 47:
            self.pwnloc -= 1
            print("capture")
            return chess.Move(47, 54)
        if self.pwnloc == 54:
            print(move_actions)
            print("promote")
            return chess.Move(54, 61, chess.KNIGHT)
        return chess.Move(self.pwnloc, self.pwnloc + 8)

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        pass

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        pass
