# Auto arborist CRISP 

In this directory, we will work on applying CRISP to the auto arborist dataset. The premise behind AA is a good match to test CRISP with: for each datum, there is an aerial image and a street level image. The goal is to classify the species of the tree based on the image pair. 

## Dataset notes 

The dataset is quite large > 1 TB I think. and it comes in a weird tfrecords format. I did some heavy processing to get the data in a suitable form. But I only did this for < 30 gb of data due to compute and storage constraints. The sample data is stored in `/mnt/disks/MOUNT_DIR/vancouver_sample/` and contains ~1000 samples from Vancouver. Each datum is a dictionary `.pkl` object file. Upon loading it in, you'll get a dict of the form: 

- image_id
- image_idx
- aerial_image
- sv_image
- lat
- lng
- label_id
- label_text
- location
- split
- og_path


See the official data card spec to learn more about each field. Though, for each datum, I added the `location`, `split`, and `og_path` fields since I plan to store all data samples in a single directory. The important fields are: 

- `aerial_image`: numpy array of the aerial image 
- `sv_image`: numpy array of the street view image 
- `label_text`: species name (i.e. the label)


### Statistics 


```
train:
prunus            192
acer              191
fraxinus           61
carpinus           57
fagus              49
magnolia           48
tilia              47
quercus            38
crataegus          27
malus              24
cornus             23
syringa            15
pyrus              14
gleditsia          13
thuja              13
aesculus            9
ulmus               5
cercidiphyllum      4
cercis              4
platanus            4
picea               2
corylus             1
---
test: 
prunus            23
acer              23
carpinus           9
fagus              7
pyrus              5
malus              5
platanus           4
cornus             3
magnolia           3
tilia              3
fraxinus           3
quercus            3
cercidiphyllum     2
corylus            2
ulmus              2
crataegus          2
gleditsia          1
picea              1
thuja              1
cercis             1
aesculus           1
syringa            1
Name: name, dtype: int64
```


### Samples 

<p align='center'>
    <img alt="picture 1" src="https://cdn.jsdelivr.net/gh/minimatest/vscode-images@main/images/c7aaca940b5c7e9daeac58e573fed997e5a77590fe3d52c4e21e608ac0955c67.png" width="350" />  
</p>
