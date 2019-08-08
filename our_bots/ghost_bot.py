from reconchess import *
import chess.engine
import os, sys
import numpy as np
import scipy.signal
import random, itertools, logging

#logging.basicConfig(level=logging.DEBUG)
STOCKFISH_ENV_VAR = 'STOCKFISH_EXECUTABLE'
UNICODE_MAP = {chess.Piece(p,c).unicode_symbol():chess.Piece(p,c).symbol() for p in range(1,7) for c in [True, False]}
#Pieces: pawn = 1, knight = 2, bishop = 3, rook = 4, queen = 5, king = 6
PIECE_VALUE_DICT = {1:1, 2:3, 3:3, 4:5, 5:9, 6:100}

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

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
		for state in self.states:
			state.turn = self.opponent_color
		if captured_my_piece:
			for state in self.states:
				attackers = state.attackers(self.opponent_color, capture_square)
				for attacker_square in attackers:
					temp = state.copy()
					move = chess.Move(attacker_square, capture_square)
					temp.push(move)
#					temp.set_piece_at(capture_square, state.piece_at(attacker_square))
#					temp.remove_piece_at(attacker_square)
					new_states.append(temp)
#			self.print_states(new_states)
		else:
			for state in self.states:
				for move in state.pseudo_legal_moves: #Assume these are all legal moves, still need to include castling
					if state.is_capture(move):
						continue
					new_state = state.copy()
					new_state.push(move)
					new_states.append(new_state)
		self.states = new_states
		print("Number of states after handling opponent move: ", len(self.states))

	def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
			Optional[Square]:
		expected = np.zeros((64,))
		for square in range(len(expected)):
			table = {}
			for state in self.states:
				piece = state.piece_at(square)
				if piece and piece.color == self.color:
					continue
				typ = piece.piece_type if piece else None
				table[typ] = table[typ] + 1 if typ in table else 1

			for piece in table:
				p = table[piece]/len(self.states)
				expected[square] += p*(len(self.states) - table[piece])

		expected = expected.reshape(8,8)
#		print(expected)
		index = 0
		maxval = -1
		for i, j in itertools.product(np.arange(1,7),np.arange(1,7)):
			val = np.sum(expected[i-1:i+2,j-1:j+2])
			if val > maxval:
				maxval = val
				index = i * 8 + j
				
#		scan = np.argmax(scipy.signal.convolve2d(np.pad(expected.reshape(8, 8), ((1,)*2,)*2, 'constant'), np.array([1]*9).reshape(3, 3)))
#		print(scan)
#		index = scan
#		index = scan if scan < 10 else scan - 10 if scan % 10 == 0 else scan - 11
#		print(index)
#		print(index)
		return index.item()

	def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
#		self.print_states()
		confirmed_states = []
		for state in self.states:
			worked = True
			for square, piece in sense_result:
				typ = piece.piece_type if piece else None
				if state.piece_type_at(square) != typ:
					worked = False
					break
			if worked:
				confirmed_states.append(state)
		self.states = confirmed_states
		print("Number of states after sensing: ", len(self.states))

	def eval(self, move, state):
		self.table[move] = self.table.get(move, 0) + (1/len(self.states))*self.eval_state(new_state)

	def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
		table = {}
		blockPrint()
		for state in self.states:
			state.turn = self.color
			for move in state.pseudo_legal_moves:
#				eval_thread = threading.Thread(target=self.eval, args=(move, new_state))
#				eval_thread.start()
				# assuming probability is constant, may change later
				sys.stderr = DevNull()
				new_state = state.copy()
				new_state.push(move)
				new_state.turn = self.opponent_color
				info = self.engine.analyse(new_state, chess.engine.Limit(time=0.010))
				if info["score"]:
					table[move] = table.get(move, 0) + (1/len(self.states))*info["score"].pov(self.opponent_color).score(mate_score=100000)
				sys.stderr = sys.__stderr__
#					continue
#				print("GOT THE INFO")
#				print(move, info["score"].pov(self.opponent_color).score())
		enablePrint()
#		print(table)
		if len(table) == 0:
			return None
		v=list(table.values())
		k=list(table.keys())
		move = k[v.index(max(v))]
		print(move)
		return move

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
		print("BET WE WON BOYS")
		self.engine.quit()
	
#	def eval_state(self, state):
#		evaluation = self.engine.analyze(state)
#		my_score = 0
#		for i in PIECE_VALUE_DICT:
#			my_score += PIECE_VALUE_DICT[i] * len(state.pieces(i,self.color))
#		opp_score = 0
#		for i in PIECE_VALUE_DICT:
#			opp_score += PIECE_VALUE_DICT[i] * len(state.pieces(i,not self.color))
#		return my_score - opp_score
	
	def print_states(self, states=None):
		if not states:
			states = self.states
		for state in states:
			print(''.join([UNICODE_MAP[x] if x in UNICODE_MAP else x for x in state.unicode()]))
			print()