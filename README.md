# About
This is the codebase for team StonkFish's bot in the Reconnaissance Blind Chess tournament held by John Hopkins University's Applied Physics Laboratory (https://rbc.jhuapl.edu)
Our team is from TJHSST.

# Requirements
This bot requires the `reconchess` and `python-chess` packages as well as many others. 
To install:
```
pip install reconchess python-chess numpy scipy
```
Or (if using pipenv):
```
pipenv install
```

This bot also requires [Stockfish](https://stockfishchess.org). Either download it and install, use `brew install stockfish` if you're on MacOS, or follow these instructions on a ssh-only Liniux terminal:

```
wget https://stockfishchess.org/files/stockfish-10-linux.zip
unzip stockfish-10-linux.zip
rm stockfish-10-linux.zip
chmod +x stockfish-10-linux.zip/Linux/stockfish_10_x64
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
 - Password: "{&aH}BhW}Xn;9MRvjMA`D6H'/"
 - Path: ~/Programs/rbc-stonkfish/our_bots/b_bot.py
 - tmux session: ras2
 - Computers: 
   - mozart 
   - brahms
   - chopin
   - ganondorf

All of the CSL computers have roughly the same specifications:
 - CPU(s): 4
 - Model name: Intel(R) Core(TM) i5-4590 CPU @ 3.30GHz
 - CPU max MHz: 3700
 - CPU min MHz: 800

### Manual Setup
  
1. ssh into remote.tjhsst.edu
2. `cd ~/Programs/rbc-stonkfish` and `git pull` to make sure the most recent code is running.
3. For each bot:
    1. ssh into the right gateway computer (ras1/ras2).
    2. For each computer:
        1. Run `tmux` to maintain the connection and then run `ssh computer`.
        2. Run `rc-connect --username username --password password bot_path --max-concurrent-games 1`.
        3. Finally, detach the tmux session (making sure not to exit) with `ctrl-b d`.
    3. After adding all the computers, run `tmux ls` and make sure there are 4 sessions.
4. After repeating the process for both bots run `top` to make sure you didn't accidentally run a `rc-connect` on the gateway computers (running high cost computations on them is a bannable offense, so be careful!).

### Automatic Setup

1. ssh into remote.tjhsst.edu
2. `cd ~/Programs/rbc-stonkfish`
3. Run `python tests/server.py A`
4. (Optional): ssh into the other gateway
5. Run `python tests/server.py B`
