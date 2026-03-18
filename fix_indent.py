with open('trading.py', 'r') as f:
    lines = f.readlines()

for i in range(789, 856):
    if lines[i].startswith('    '):
        lines[i] = lines[i][4:]

with open('trading.py', 'w') as f:
    f.writelines(lines)
