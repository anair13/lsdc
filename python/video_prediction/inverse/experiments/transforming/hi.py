import glob

def find_repeats(dir):
    """list of lists [[foo, bar], []]"""
    hashes = {}
    for folder in glob.glob('dir/*'):
        for file in glob.glob('folder/*'):
            f = open(file, "r")
            c = f.read()
            h = hash(c)
            if h in hashes:
                hashes[h].append(file)
            else:
                hashes[h] = [file]

    return hashes
