import sys

def parse_pd_main_canvas(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    objects = []
    connections = []
    
    in_subpatch = 0
    
    for line in lines:
        if line.startswith('#N canvas'):
            in_subpatch += 1
            if in_subpatch == 1:
                objects.append(line.strip()) # Main canvas obj
        elif line.startswith('#X restore'):
            in_subpatch -= 1
        elif in_subpatch == 0:
            if line.startswith('#X obj') or line.startswith('#X msg') or line.startswith('#X floatatom') or line.startswith('#X text') or line.startswith('#X tgl') or line.startswith('#X bng'):
                objects.append(line.strip())
            elif line.startswith('#X connect'):
                parts = line.split()
                if len(parts) >= 6:
                    connections.append((int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5].replace(';', ''))))
                
    return objects, connections

objects, connections = parse_pd_main_canvas('c:/Users/user/Desktop/파이썬/Cosmos/cosmos_sound/cosmos_sound_2.pd')
for i, obj in enumerate(objects):
    print(f"{i}: {obj}")

print("\nConnections:")
for c in connections:
    print(f"From {c[0]} ({objects[c[0]] if c[0] < len(objects) else '?'}) outlet {c[1]} to {c[2]} ({objects[c[2]] if c[2] < len(objects) else '?'}) inlet {c[3]}")
