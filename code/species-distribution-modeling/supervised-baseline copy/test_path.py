# print out data dir to ensure path is correct 

import os
import sys
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='List the contents of a directory. Specify --drive of on colab.')
parser.add_argument('--drive', action='store_true', help='Are you on Colab? If provided, sets drive to False, otherwise True')
args = parser.parse_args()

# Determine value of drive based on flag
if args.drive:
    drive = True
else:
    drive = False



# Use drive value to list directory contents
if drive:
    path = "/content/drive/MyDrive/CS 197 Research Team 3/data/"  # google drive string 
else:
    path = "../../../data/playground"  # local repo path 

with os.scandir(path) as entries:
    for entry in entries:
        print(entry.name)

