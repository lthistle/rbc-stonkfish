from reconchess.scripts import rc_bot_match
from reconchess import *
import datetime

TEST_GAMES = 30
#TEST_BOT1 = '../sample_bots/trout_bot.py'
TEST_BOT1 = '../our_bots/experimental_bot.py'
TEST_BOT2 = '../our_bots/dontdie.py'
TEST_NAME1 = 'Experimental Bot'
TEST_NAME2 = 'Defensive Bot'

def run_game(swap):
    _, bot1 = load_player(TEST_BOT1)
    _, bot2 = load_player(TEST_BOT2)
    
    bot1 = bot1()
    bot2 = bot2()

    players = [bot1, bot2]
    player_names = [TEST_NAME1, TEST_NAME2]
    if swap:
        players.reverse()
        player_names.reverse()

    colors = ["white", "black"]
    game = LocalGame(900)
    winner_color, win_reason, history = play_local_game(players[0], players[1], game)
    winner = 'Draw' if winner_color is None else chess.COLOR_NAMES[winner_color]
    
    winner_name = player_names[colors.index(winner)]
    
    print('Game Over!')
    print('Winner: {} as {} because of {}!'.format(winner_name, winner, win_reason))

    timestamp = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

    replay_path = '{}-{}-{}-{}.json'.format(player_names[0], player_names[1], winner, timestamp)
    print('Saving replay to {}...'.format(replay_path))
    history.save(replay_path)
    return winner_name

winners = {}
for i in range(TEST_GAMES):
    winner = run_game(False)
    print(winner)
    winners[winner] = winners.get(winner, 0) + 1
    print(winners)
