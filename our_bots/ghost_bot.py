from reconchess import *
import chess.engine
import os, sys
import numpy as np
import scipy.signal
import random, itertools, logging, time
from scipy.interpolate import interp1d

#logging.basicConfig(level=logging.DEBUG)
STOCKFISH_ENV_VAR = 'STOCKFISH_EXECUTABLE'
UNICODE_MAP = {chess.Piece(p,c).unicode_symbol():chess.Piece(p,c).symbol() for p in range(1,7) for c in [True, False]}

MIN_TIME = 5
MAX_TIME = 30
MAX_MOVE_COUNT = 12000

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
        self.first_turn = True

	def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        if self.color and self.first_turn:
            self.first_turn = False
            return

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
#			self.print_states(new_states)
		else:
			for state in self.states:
				for move in state.pseudo_legal_moves: #Assume these are all legal moves, still need to include castling
                    if not state.is_capture(move):
                        new_state = state.copy()
    					new_state.push(move)
                        if str(new_state) not in states_set:
                            states_set.add(str(new_state))
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

		print("Number of moves to analyze", move_count)

        time_func = interp1d([1,MAX_MOVE_COUNT],[MIN_TIME,MAX_TIME])
		time_to_analyze = time_func(min(move_count, MAX_MOVE_COUNT))
        time_per_move = time_to_analyze / move_count
        print(time_to_analyze, "seconds")

        for state in self.states:
			state.turn = self.color
			for move in state.pseudo_legal_moves:
				sys.stderr = DevNull()

                if move.to_square == state.king(self.opponent_color):
                    points = WIN
				else:
					temp = state.copy()
					temp.push(move)
					temp.turn = self.opponent_color
					info = self.engine.analyse(temp, chess.engine.Limit(time=time_per_move))
					points = info["score"].pov(self.color).score(mate_score=MATE)
				if info["score"]:
					table[move] = table.get(move, 0) + points/len(self.states)
				sys.stderr = sys.__stderr__

        if len(table) == 0: return None
		v = -float("inf")
        best = max(table, key=lambda move: table[move])
		print(best, table[best])
		print("Time left before starting calculations for current move: ", seconds_left/60)
		print("Time left now: ", (seconds_left - time.time() + tim)/60)
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
			print(''.join([UNICODE_MAP.get(x,x) for x in state.unicode()]) + "\n")