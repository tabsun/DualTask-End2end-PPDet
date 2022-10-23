import os
import json
import subprocess
import cv2
import sys

t = cv2.getTickCount()
ps = []
process_num = 16
for pid in range(process_num):
    cmd = "python single_process.py %s %s %d %d" % (sys.argv[1], sys.argv[2], pid, process_num)
    p = subprocess.Popen(list(cmd.split(' ')))
    ps.append(p)

for p in ps:
    p.wait()

whole_str ='{"result": [' 
for pid in range(process_num):
    cur_file = '%s_%d.json'%(sys.argv[2], pid)
    data = open(cur_file, 'r').readlines()[0][1:-1]
    whole_str += data
    if(pid < process_num-1):
        whole_str += ','
    os.remove(cur_file)
whole_str += ']}'
    
with open(sys.argv[2], 'w') as f:
    f.write(whole_str)

t = cv2.getTickCount() - t
print("All use time %gs" % (t/cv2.getTickFrequency()))
