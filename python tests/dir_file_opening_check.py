import os
import io
from os import path

for i in range(10):
    if not os.path.exists('dir/'):
        os.mkdir('dir/')
        input('Press Enter to continue')
    with open(f'dir/file{i}', 'w'):
        pass
