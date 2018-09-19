"""
    Moves a sample of a dataset to a new subdirectory
"""

import os
import sys
import shutil
import random

"""
n = number of poses to sample
dir1 = entire dataset directory
dir2 = destination directory for dataset sample
"""

def sampler(n, dir1, dir2):

    """ getting n sdf files from the entire dataset """
    files = []
    for filename in os.listdir(dir1):
        if filename[0] != '.' and not os.path.isdir(filename[0]) and filename.split('.')[1] == 'sdf':
            files.append(filename)


    assert (n <= len(files)), 'Cannot request more files than in directory.'

    random.shuffle(files)

    """ copying sample dataset to destination directory """
    for i in range(n):
        shutil.copyfile(os.path.join(dir1, files[i]) , os.path.join(dir2, files[i]))

if __name__ == "__main__":
    sampler(int(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]))