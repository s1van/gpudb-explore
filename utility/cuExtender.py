#!/usr/bin/python

import sys, getopt, string, tempfile, re

def usage():
	print "Usage: cuExtender.py [--help] [--ref=referenceFile] [--src=cuSource] [--dump=outFile]"
	print """Reference Format: FunctionPattern\tARG1\tREF\tARG2\tREF..."""

def main():
	if len(sys.argv) < 2:
		print "not enough arguments"
		usage()
		sys.exit()
	
	try:
		opts, args = getopt.getopt(sys.argv[1:], "hr:s:d:", ["help", "ref=", "src=", "dump="])
	except getopt.GetoptError:
		usage()
		sys.exit(2)

	noDumpFile= True
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			usage()
			sys.exit()
		elif opt in ("-r", "--ref"):
			refFile = open(arg, 'r')
		elif opt in ("-s", "--src"):
			srcFile = open(arg, 'r')
		elif opt in ("-d", "--dump"):
			dumpFile = open(arg, 'r')
			noDumpFile = False
		else:
			print "unhandled option"
			usage()
			sys.exit()

	if noDumpFile:
		dumpFile = sys.stdout
	
	#parse the refFile
	refs = {}
	for line in refFile:
		elems = line.strip('\n').split("\t");
		i = 0
		pattern = elems[0]
		refs[pattern] = {}
		for i in range(1,len(elems),2): 
			refs[pattern][elems[i]] = elems[i+1]
	
	#print refs.keys()
	for line in srcFile:
		if (re.search('\<\<\<', line) != None):	#must be a kernel call
			patternFound = False
			for pattern in refs.keys():
				if (re.search('[ |\t]' + pattern + '\<', line) != None):
					spaces = line[0:(len(line)-len(line.lstrip()))]
					patternFound = True
					ref = refs[pattern]
					print spaces + 'do{'
					for pos in ref.keys():
						print spaces + '\t' + "cudaReference(" + pos + ", " + ref[pos] + ");"
					print '\t' + line.strip('\n')
					print spaces + '} while(0);'
			if (not patternFound):
				print line.strip('\n')
		else:
			print line.strip('\n')
	

if __name__ == "__main__":
    main()
