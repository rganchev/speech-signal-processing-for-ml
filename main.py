import importlib
import sys
import os
from pathlib import Path
import random

commands_dir = Path('src').joinpath('commands').absolute()
commands = [
  os.path.splitext(f)[0]
    for f in os.listdir(commands_dir)
    if os.path.isfile(commands_dir.joinpath(f)) and os.path.splitext(f)[1] == '.py'
]

def usage():
  print('\nUsage:\n\tpython main.py <command>\n\nWhere <command> is one of %s\n' % commands)
  exit(1)

if len(sys.argv) != 2:
  usage()

_, command = sys.argv
if command not in commands:
  usage()

module = importlib.import_module('.commands.%s' % command, 'src')

random.seed(54321)
module.main()