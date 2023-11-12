import os
import argparse

parser = argparse.ArgumentParser(description='Kill process.')
parser.add_argument("--file",            
                     default="", 
                     type=str, 
                     help="File to ps -ef|grep python|grep sbmaruf Value: (str)")
parser.add_argument("--search-tag",            
                     default="python3 -u src/main.py", 
                     type=str, 
                     help="The X tag that will be searched in `ps -ef|grep python|grep X`")
parser.add_argument("--ignore-safety",            
                     default=0, 
                     type=int, 
                     help="Ignore safety lookup, just kill all the matched process.")
parser.add_argument("--dry-run",            
                     action='store_true',
                     help="Just print the query process, do not perform any operation.")
parser.add_argument("--use-sudo",            
                     action='store_true',
                     help="Just print the query process, do not perform any operation.")
params = parser.parse_args()
cmd = "ps -ef|grep python|grep \"{}\" > out.txt".format(params.search_tag)
print("Executing {}".format(cmd))
os.system(cmd)

flag = 0
if params.file == '':
    flag = 1
    params.file = "out.txt"

with open(params.file, "r") as filePtr:
    for line in filePtr:
        line = line.strip()
        if "kill-process.py" in line or "grep" in line:
            continue
        print("Process Information :: {}".format(line))
        if params.dry_run == True:
            continue
        line = line.split()
        process2 = line[2]
        process1 = line[1]
        cmd = "kill -9 {}".format(process2)
        ret = 'y'
        if params.ignore_safety == 0:
            ret = str(input("Do you want to kill :"))
        if ret == "y":
            os.system(cmd)
            cmd = "kill -9 {}".format(process1)
            if params.use_sudo:
                cmd = f"sudo {cmd}"
            os.system(cmd)
        else:
            pass

if os.path.exists("out.txt"):
    os.system("rm out.txt")