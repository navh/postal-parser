filenames = ['conll14.txt', 'conll15.txt', 'conll16.txt', 'conll57.txt']
with open('can_us_conll.txt', 'a') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())