from reconchess import *
import chess

#not currently compatible with reconchess/chess types
#just for testing
convert = {'a1': 0, 'a2': 8, 'a3': 16, 'a4': 24, 'a5': 32, 'a6': 40, 'a7': 48, 'a8': 56, 'b1': 1, 'b2': 9, 'b3': 17, 'b4': 25, 'b5': 33, 'b6': 41, 'b7': 49, 'b8': 57, 'c1': 2, 'c2': 10, 'c3': 18, 'c4': 26, 'c5': 34, 'c6': 42, 'c7': 50, 'c8': 58, 'd1': 3, 'd2': 11, 'd3': 19, 'd4': 27, 'd5': 35, 'd6': 43, 'd7': 51, 'd8': 59, 'e1': 4, 'e2': 12, 'e3': 20, 'e4': 28, 'e5': 36, 'e6': 44, 'e7': 52, 'e8': 60, 'f1': 5, 'f2': 13, 'f3': 21, 'f4': 29, 'f5': 37, 'f6': 45, 'f7': 53, 'f8': 61, 'g1': 6, 'g2': 14, 'g3': 22, 'g4': 30, 'g5': 38, 'g6': 46, 'g7': 54, 'g8': 62, 'h1': 7, 'h2': 15, 'h3': 23, 'h4': 31, 'h5': 39, 'h6': 47, 'h7': 55, 'h8': 63}
rows, cols = '12345678', 'abcdefgh'
valid_squares = {letter + number for letter in cols for number in rows}
color = chess.WHITE
boardlist = [chess.Board()]
seenboards =  {boardlist[0].board_fen()}
#takes dict of format {'square_uci': chess.PIECE_TYPE} (piece_type is None if empty square)
def filter_boards(known_locations):
    global boardlist
    good_boards = []
    for board in boardlist:
        board_works = True
        for square in known_locations:
            expected_type = known_locations[square]
            check_piece = board.piece_at(convert[square])
            if check_piece is None and expected_type is not None:
                    board_works = False
                    break
            if check_piece is not None and expected_type != check_piece.piece_type:
                    board_works = False
                    break
        if board_works:
            good_boards.append(board)
    boardlist = good_boards


#takes boolean and string of format 'square_uci'
def handle_opponent_move(piece_was_captured, capture_location):
    pass

#takes string of format 'square_uci'
def sense_area(center_square):
    global cols, rows
    squares = {}
    col, row = list(center_square)
    clist, rlist = [col], [row]
    a, b = cols.index(col) - 1, cols.index(col) + 1 #someone should find a better way to compute the square ucis
    if a >= 0: clist.append(cols[a])
    if b < 8: clist.append(cols[b])
    a, b = rows.index(row) - 1, rows.index(row) + 1
    if a >= 0: rlist.append(rows[a])
    if b < 8: rlist.append(rows[b])
    sensed = dict()
    for c in clist:
        for r in rlist:
            sensed[c + r] = eval(input('piece type at {}? '.format(c + r)))
    return sensed

#take dict of format {'square_uci': 'piece_type'}
def handle_sense(pieces_sensed):
    filter_boards(pieces_sensed)

def make_move(start_sq, goal_sq):
    pass

def handle_move(move_was_successful, different_move_location):
    pass

def calc_next_boards():
    global boardlist
    global seenboards
    for x in range(len(boardlist)):
        board = boardlist[x]
        for move in board.legal_moves:
            new = board.copy()
            new.push(move)
            new.turn = chess.WHITE
            if new.board_fen() in seenboards: continue
            seenboards.add(new.board_fen())
            boardlist.append(new)


#testing stuff plz ignore uwu
def blp(): global boardlist; print("Boardlist length: {}".format(len(boardlist)))

blp()
calc_next_boards()
blp()
calc_next_boards()
blp()

calc_next_boards()
blp()
exit()
calc_next_boards()
blp()
sense = sense_area('b2')
handle_sense(sense)
calc_next_boards()
blp()
sense = sense_area('e2')
handle_sense(sense)
blp()

exit()
calc_next_boards()
print("Boardlist length: {}".format(len(boardlist)))
filter_boards({'a1':chess.ROOK, 'b1':chess.KNIGHT, 'c1':chess.BISHOP, 'a2':chess.PAWN, 'b2':chess.PAWN, 'c2':chess.PAWN, 'a3':None, 'b3':None, 'c3':None})
print("Boardlist length: {}".format(len(boardlist)))


{'a1':chess.ROOK, 'b1':chess.KNIGHT, 'c1':chess.BISHOP, 'a2':chess.PAWN, 'b2':chess.PAWN, 'c2':chess.PAWN, 'a3':None, 'b3':None, 'c3':None}