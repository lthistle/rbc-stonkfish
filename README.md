# About
This is the codebase for team t2j1's bot in the Reconnaissance Blind Chess tournament held by John Hopkins University's Applied Physics Laboratory (https://rbc.jhuapl.edu)
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
