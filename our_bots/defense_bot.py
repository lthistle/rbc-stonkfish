import random
from reconchess import *
import chess.engine
import os
import numpy as np
import scipy.signal
def fetch_fen(board, opp_color):
    piece_strs = []
    pawns = str(int(board.pieces(1, opp_color)))
    knights = str(int(board.pieces(2, opp_color)))
    bishops = str(int(board.pieces(3, opp_color)))
    rooks = str(int(board.pieces(4, opp_color)))
    queens = str(int(board.pieces(5, opp_color)))
    king = str(int(board.pieces(6, opp_color)))
    can_castle = str(int(board.castling_rights))
    div = '.'
    return pawns + div + knights + div + bishops + div + \
    rooks + div + queens + div + king + div + can_castle

convert = {'a1': 0, 'a2': 8, 'a3': 16, 'a4': 24, 'a5': 32, 'a6': 40, 'a7': 48, 'a8': 56, 'b1': 1, 'b2': 9, 'b3': 17, 'b4': 25, 'b5': 33, 'b6': 41, 'b7': 49, 'b8': 57, 'c1': 2, 'c2': 10, 'c3': 18, 'c4': 26, 'c5': 34, 'c6': 42, 'c7': 50, 'c8': 58, 'd1': 3, 'd2': 11, 'd3': 19, 'd4': 27, 'd5': 35, 'd6': 43, 'd7': 51, 'd8': 59, 'e1': 4, 'e2': 12, 'e3': 20, 'e4': 28, 'e5': 36, 'e6': 44, 'e7': 52, 'e8': 60, 'f1': 5, 'f2': 13, 'f3': 21, 'f4': 29, 'f5': 37, 'f6': 45, 'f7': 53, 'f8': 61, 'g1': 6, 'g2': 14, 'g3': 22, 'g4': 30, 'g5': 38, 'g6': 46, 'g7': 54, 'g8': 62, 'h1': 7, 'h2': 15, 'h3': 23, 'h4': 31, 'h5': 39, 'h6': 47, 'h7': 55, 'h8': 63}

class DefenseBot(Player):
    def __init__(self):
        STOCKFISH_ENV_VAR = 'STOCKFISH_EXECUTABLE'
        stockfish_path = os.environ[STOCKFISH_ENV_VAR]
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.color, self.opponent = None, None
        self.possible_boards = []
        self.piece_locations = {x: set() for x in range(1, 7)}
        self.held_squares = set()
    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.color = color
        self.opponent = not color
        self.possible_boards = [board]
        self.seen_boards = {fetch_fen(board, self.opponent)}
        all_pieces = board.piece_map()
        for sqindex in all_pieces:
            piece = all_pieces[sqindex]
            if piece.color == self.color:
                self.piece_locations[piece.piece_type].add(sqindex)
                self.held_squares.add(sqindex)

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        self.plot_opp_moves()
        print("Before handling opponent's move: {}".format(len(self.possible_boards)))
        if captured_my_piece:
            self.filter_opp_loc({capture_square : captured_my_piece})
        print("After handling opponent's move: {}".format(len(self.possible_boards)))

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[Square]:
        expected = np.zeros((64,))
        for square in range(64):
            table = {}
            for state in self.possible_boards:
                piece = state.piece_at(square)
                if piece is None or piece.color == self.opponent:
                    typ = piece.piece_type if piece is not None else None
                    table[typ] = table.get(typ, 0) + 1
            for piece in table:
                expected[square] += table[piece]*(len(self.possible_boards) - table[piece])/len(self.possible_boards)
        return np.argmax(scipy.signal.convolve2d(expected.reshape(8, 8), np.array([1]*9).reshape(3, 3))[1:-1, 1:-1])

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        print("Before handling sense result: {}".format(len(self.possible_boards)))
        scan_result = dict()
        for sq, piece in sense_result:
            scan_result[sq] = piece
        print(scan_result)
        self.filter_by_pieces(scan_result)
        print("After handling sense result: {}".format(len(self.possible_boards)))

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        return random.choice(move_actions + [None])

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move], captured_opponent_piece: bool, capture_square: Optional[Square]):
        print("Before handling move result: {}".format(len(self.possible_boards)))
        if requested_move is not None:
            self.filter_legal_move(requested_move, requested_move == taken_move)
        if captured_opponent_piece:
            self.filter_opp_loc({capture_square: True})
        if taken_move is not None:
            self.make_move(taken_move)
        else:
            for b in self.possible_boards:
                b.turn = self.opponent
        print("After handling move result: {}".format(len(self.possible_boards)))

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        if winner_color == self.color:
            print('Game Won')
        else:
            print('Game Lost ')
    
    def make_move(self, move):
        for b in self.possible_boards:
            assert b.turn == self.color
            b.push(move)

    def plot_opp_moves(self):
        for x in range(len(self.possible_boards)):
            b = self.possible_boards[x]
            assert b.turn == self.opponent
            for move in b.pseudo_legal_moves:
                new_b = b.copy()
                new_b.push(move)
                fen = fetch_fen(new_b, self.opponent)
                if fen in self.seen_boards: continue
                self.seen_boards.add(fen)
                self.possible_boards.append(new_b)
            b.turn = self.color 
        #print('after plotting...')
        #for nb in self.possible_boards:
        #    if nb.turn 

        
    def filter_by_presence(self, squares): #takes dict like {14:True, 12:False} where bool refers to if a piece should be there
        good_boards = []
        for b in self.possible_boards:
            b_works = True
            for sq in squares:
                piece_present = b.piece_at(sq)
                occupied = True if piece_present is not None else False
                expected = squares[sq]
                if occupied != expected:
                    b_works = False
                    break
            if b_works:
                good_boards.append(b)
        self.possible_boards = good_boards

    def filter_opp_loc(self, squares): #True: expect opponent, False: don't expect opponent
        good_boards = []
        for b in self.possible_boards:
            b_works = True
            for sq in squares:
                color_present = b.color_at(sq)
                is_opp = True if color_present == self.opponent else False
                expected = squares[sq]
                if expected == True and is_opp == False:
                    b_works = False
                    break
                if expected == False and is_opp == True:
                    b_works = False
                    break
            if b_works:
                good_boards.append(b)
        self.possible_boards = good_boards
    
    def filter_by_pieces(self, squares): #{id: Piece}, where piece can be None
        good_boards = []
        for b in self.possible_boards:
            b_works = True
            for sq in squares:
                expected_type = squares[sq]
                check_piece = b.piece_at(sq)
                if check_piece is None and expected_type is not None:
                        b_works = False
                        break
                if check_piece is not None and expected_type is not None and expected_type.piece_type != check_piece.piece_type:
                        b_works = False
                        break
            if b_works:
                good_boards.append(b)
        self.possible_boards = good_boards

    def filter_legal_move(self, move, is_legal):
        good_boards = []
        for b in self.possible_boards:
            if (move in b.pseudo_legal_moves) == is_legal:
                good_boards.append(b)
        self.possible_boards = good_boards
db = DefenseBot()
b = chess.Board()
db.handle_game_start(chess.BLACK, b, 'lel')
