import sys

def parse_pd(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    objects = []
    connections = []
    for line in lines:
        if line.startswith('#X obj') or line.startswith('#X msg') or line.startswith('#X floatatom') or line.startswith('#X text') or line.startswith('#X ncanvas') or line.startswith('#X tgl') or line.startswith('#X bng'):
            objects.append(line.strip())
        elif line.startswith('#X connect'):
            parts = line.split()
            if len(parts) >= 6:
                connections.append((int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5].replace(';', ''))))
                
    return objects, connections

objects, connections = parse_pd('c:/Users/user/Desktop/파이썬/Cosmos/cosmos_sound/cosmos_sound_2.pd')
for i, obj in enumerate(objects):
    print(f"{i}: {obj}")

print("\nConnections to DAC:")
dac_indices = [i for i, obj in enumerate(objects) if 'dac~' in obj]
for dac_idx in dac_indices:
    for c in connections:
        if c[2] == dac_idx:
            print(f"From {c[0]} (outlet {c[1]}) to {c[2]} (inlet {c[3]})")
