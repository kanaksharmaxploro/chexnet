import re

with open('ChexnetTrainer.py', 'r') as f:
    content = f.read()

# Catch any spacing variation: async=True, async = True, async =True, etc.
content = re.sub(r'async\s*=\s*True', 'non_blocking=True', content)

with open('ChexnetTrainer.py', 'w') as f:
    f.write(content)

print("Done")