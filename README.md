# About
This is the codebase for team StonkFish's bot in the Reconnaissance Blind Chess tournament held by John Hopkins University's Applied Physics Laboratory (https://rbc.jhuapl.edu)
Our team is from TJHSST.

# Requirements
This bot requires the `reconchess` and `python-chess` packages. 
To install:
```
pip install reconchess python-chess
```
Or (if using pipenv):
```
pipenv install
```

# Setup
Set an environmental variable called `STOCKFISH_EXECUTABLE` pointing to the location of the stockfish executable. In terminal:
```
export STOCKFISH_EXECUTABLE=/path_to_stockfish
```
## Tournament Setup

"A" bot 
 - Username: StonkFish
 - Password: "tj.rbc2021Gang"
 - Path: ~/Programs/rbc-stonkfish/our_bots/ghost_bot.py
 - tmux session: ras1
 - Computers: 
    - mario 
    - luigi
    - peach
    - yoshi

"B" bot 
 - Username: HestiaBestia
 - Password: "{&aH}BhW}Xn;9MRvjMA\`D6H'/"
 - Path: ~/Programs/rbc-stonkfish/our_bots/b_bot.py
 - tmux session: ras2
 - Computers: 
    - mozart 
    - brahms
    - chopin
    - ganondorf
    
ssh into remote.tjhsst.edu and make sure you are in the right gateway computer.

`cd ~/Programs/rbc-stonkfish` and `git pull` to make sure the most recent code is running.

Run `tmux` to maintain the connection and then run `ssh computer`.

Run `rc-connect --username username --password password bot_path --max-concurrent-games 1`.

Finally, detach the tmux session (making sure not to exit).

Run `tmux ls` and make sure there are 4 sessions.

After repeating the process for both bots run `top` to make sure you didn't accidently run a `rc-connect` on the gateway computers (ras1 and ras2).
