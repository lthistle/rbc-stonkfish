from reconchess import *
import datetime

STOCKFISH_ENV_VAR = 'STOCKFISH_EXECUTABLE'
LEELA_ENV_VAR = 'LEELA_EXECUTABLE'

def run_game(swap):
    _, trout = load_player('../sample_bots/trout_bot.py')
    leela = trout(LEELA_ENV_VAR)
    stockfish = trout(STOCKFISH_ENV_VAR)

    players = [leela, stockfish]
    player_names = ["Leela", "Stockfish"]
    if swap:
        players.reverse()
        player_names.reverse()

    colors = ["white", "black"]
    game = LocalGame(90)
    winner_color, win_reason, history = play_local_game(players[0], players[1], game)
    winner = 'Draw' if winner_color is None else chess.COLOR_NAMES[winner_color]
    
    winner_name = player_names[colors.index(winner)]
    
    print('Game Over!')
    print('Winner: {} as {} because of !'.format(winner_name, winner, win_reason))

    timestamp = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

    replay_path = '{}-{}-{}-{}.json'.format(player_names[0], player_names[1], winner, timestamp)
    print('Saving replay to {}...'.format(replay_path))
    history.save(replay_path)
    return winner_name

winners = {}
for i in range(100):
    winner = run_game(True if i%2 == 0 else False)
    print(winner)
    winners[winner] = winners.get(winner, 0) + 1
    print(winners)