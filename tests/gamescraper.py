from bs4 import BeautifulSoup
from reconchess.history import GameHistory
import urllib.request
import json
import re
import yaml
import chess

game_id = input("Game ID? ")
output_file = input("Output File? ")
page = urllib.request.urlopen("https://rbc.jhuapl.edu/games/" + game_id)
soup = BeautifulSoup(page, 'html.parser')

gh = GameHistory()

a_tags = []
for a in soup.find_all('a'):
    if 'users' in a.get('href'):
        a_tags.append(a.text)   

gh._white_name, gh._black_name = a_tags[1:3]
if a_tags[3] == gh._white_name:
    gh._winner_color = chess.WHITE
else:
    gh._winner_color = chess.BLACK
gh._win_reason = {"type": "WinReason", "value": "KING_CAPTURE"}

tag_content = soup.find_all('script')[-1].string
matches = re.findall("var\ actions\ \=([\s\S]*?);", tag_content)
game_info = yaml.load(matches[0], Loader=yaml.FullLoader)

for action in game_info:
    color = action['turn_color']
    if action['phase'] == 'sense':
        gh._senses[color].append(action['sense'])
        gh._sense_results[color].append(action['sense_result'])
        gh._fens_before_move[color].append(action['fen'])
    else:
        gh._requested_moves[color].append(action['requested_move'])
        gh._taken_moves[color].append(action['taken_move'])
        gh._capture_squares[color].append(action['capture_square'])
        gh._fens_after_move[color].append(action['fen'])

gh.save(output_file)