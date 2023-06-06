import pdb 
import data 
import os

# params 
s3_bucket = 'dummynaturalist'
csv_path = "filtered.csv"
images_dir_path = "images"
remote_sensing_dir_path = "remote_sesning"

ds = data.S3CrispDataset(s3_bucket, csv_path, images_dir_path, remote_sensing_dir_path)
gl, rs = ds[5]

