def get_fen(board):
    return board.board_fen()
def get_fen_v2(board, opp_color):
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

import time
import chess
epoch = 100000
b = chess.Board()
t0 = time.time()
s1 = set()
for x in range(epoch):
    s1.add(get_fen(b))
t1 = time.time()
s2 = set()
for x in range(epoch):
    s2.add(get_fen_v2(b))
t2 = time.time()

print(get_fen_v2(b))
print("Running {} iterations...".format(epoch))
print('get_fen finished in {}'.format(t1 - t0))
print('get_fen_v2 finished in {}'.format(t2 - t1))
