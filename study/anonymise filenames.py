import os

import hashlib

def md5(txt):
    return hashlib.md5(txt.encode()).hexdigest()

def genName(fileName):
    return f"renamed_{md5(fileName)}.wav"

def generateMappings(dirs):
    out = []
    for d in dirs:
        for f in os.listdir(d):
            if f[-4:] == ".wav":
                out.append((d + f, d + genName(f)))
    return out

def replaceAll(string,replacements):
    for (old,new) in replacements:
        string = string.replace(old,new)
    return string

def replaceInFile(fname):
    with open(fname, "rt") as fin:
        with open("replaced.txt", "wt") as fout:
            for line in fin:
                fout.write(replaceAll(line,mappings))

def writeOutMappings():
     with open("mappings.txt", "wt") as fout:
            for mapping in mappings:
                fout.write(f"{mapping[0]} -> {mapping[1]}\n")

def renameFiles():
    for (old,new) in mappings:
        #os.rename(old, new)
        print(f"Renamed {old} to {new}")

dirs = ["./wavs/sec1/","./wavs/sec2/"]
mappings = generateMappings(dirs)
replaceInFile("input.txt")
renameFiles()
writeOutMappings()
