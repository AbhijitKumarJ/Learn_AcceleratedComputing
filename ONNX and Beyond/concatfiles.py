import time, glob

outfilename = "all.md"

filenames = glob.glob('*.md')

with open(outfilename, 'w', encoding="utf8") as outfile:
    for fname in filenames:
        with open(fname, 'r', encoding="utf8") as readfile:
            outfile.write("File: " + fname)
            outfile.write("\n\n\n")
            infile = readfile.read()
            for line in infile:
                outfile.write(line)
            outfile.write("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")