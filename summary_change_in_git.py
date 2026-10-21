#! /usr/bin/env python
import os
import pandas as pd

unique_files=[]
large_files=[]
total_size=0
os.system("git status > change_in_git")
os.system("git add -n . > git_to_be_added")
with open("change_in_git") as f:
    lines=f.readlines()
    for line in lines:
        filename=line.split("\t")[-1].replace("\n","")
        #print(filename)
        if os.path.isfile(filename):
            megabyte=os.stat(filename).st_size / (1024 * 1024)
            total_size+=megabyte
            #print(megabyte)
            if megabyte >20:
                raise ValueError(f"{filename}larger than 20" )
  
data=[]
with open("git_to_be_added") as f:
    lines=f.readlines()
    for line in lines:
        filename=line.replace("add '","").replace("'","").replace("\n","")
        #print(filename)
        if os.path.isfile(filename):
            
            megabyte=os.stat(filename).st_size / (1024 * 1024)
            total_size+=megabyte
            print(filename,":",megabyte,"Mb")
            data.append([filename,megabyte])
            if megabyte >20:
                raise ValueError(f"{filename} larger than 20" )
                large_files.append(filename)
if len(large_files)>0:
    raise ValueError(large_files)
if total_size>49:
    df=pd.DataFrame(data,columns=["filename","size"])
    df.to_csv("filesize.csv")
    raise ValueError("commit larger than 49M may cause problem")      
else:
    print(f"check done, all is not large file\n commit size is {total_size:.2f} Mb and is smaller than 49Mb")
