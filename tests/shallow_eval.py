import reconchess
import chess

piece_vals = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 100}

#get some possible game boards
b = chess.Board('r2q1rk1/pb1nbppp/np2p3/P1p1N3/2P5/3P2P1/1B1NPPBP/R2Q1RK1 b - - 4 13')
test_boards = []
for move in b.legal_moves:
    copy = b.copy()
    copy.push(move)
    test_boards.append(copy)


def score_board(b):
    score = 0
    for move in b.legal_moves:
        dest = move.to_square
        p = b.piece_at(dest)
        if p is not None:
            score += piece_vals[p.piece_type]
    return score

import time
t0 = time.time()
scores = []
for x in test_boards:
    scores.append(score_board(x))

print( (time.time() - t0) / len(test_boards) )
print(scores)