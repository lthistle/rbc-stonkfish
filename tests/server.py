import os, sys, subprocess, json

subprocess.run(["tmux", "kill-server"])
subprocess.run(["git", "pull"])

bot = json.load(open("bots.json"))[sys.argv[1]]
# create temp file becuase bash is stupid with the ` character
with open("temp.sh", "w") as f:
    f.write(f"~/.local/bin/rc-connect --username {bot['username']} --password \"{bot['password']}\" {bot['path']} --max-concurrent-games 1")

for computer in bot["computers"]:
    subprocess.run(["tmux", "new-session", "-d", "-s", computer, f"ssh {computer} bash -s < temp.sh"])

os.remove("temp.sh")
