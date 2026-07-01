import subprocess

# Read the file with the bash commands
with open('exp3-download_data-14000.sh', 'r') as file:
    commands = file.readlines()

total_commands = len(commands)

# Execute each command and print progress
for i, command in enumerate(commands, start=1):
    print(f"Executing {i}/{total_commands}: {command.strip()}")
    subprocess.run(command, shell=True)

print("All commands executed.")
