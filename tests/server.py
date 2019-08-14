import os, sys, subprocess, json

# number of CPU cores on the computer
# has to be at least 4 for an individual computer
CORES = 4

subprocess.run(["tmux", "kill-server"])
subprocess.run(["git", "pull"])

bot = json.load(open("tests/bots.json"))[sys.argv[1]]
print(f"Loaded bot {bot['username']}")

# create temp file becuase bash is stupid with the ` character
with open("temp.sh", "w") as f:
    f.write("export STOCKFISH_EXECUTABLE=~/stockfish-10-linux/Linux/stockfish_10_x64 \n")
    f.write("pkill rc-connect \n")
    f.write(f"~/.local/bin/rc-connect --username {bot['username']} --password \"{bot['password']}\" {bot['path']} --max-concurrent-games {CORES}")

for computer in bot["computers"]:
    if len(sys.argv) > 2 and sys.argv[2] == "kill":
        print(f"Killing existing connections on {computer}")
        subprocess.run(["ssh", computer, "pkill rc-connect"])
    else:
        print(f"Starting new connection on {computer}")
        subprocess.run(["tmux", "new-session", "-d", "-s", computer, f"ssh {computer} bash -s < temp.sh"])

os.remove("temp.sh")
