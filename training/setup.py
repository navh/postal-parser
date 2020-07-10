
# Archi & Ian
FILE_NAME = '../data/sample_of_toronto.csv'
OUT_FILE_NAME = '../data/CoNLL_addresses.txt'

"""
Example of .csv file:

LON,    LAT,    NUMBER, STREET,     UNIT,   CITY,       DISTRICT,   REGION, POSTCODE,   ID, HASH
-49.41, 29.523, 52,     Main St.,   3a,     Toronto,    ,           ON,     N6C4E9,     ,   529a2b19a...
-50.1,  -23.1,  9,      South St.,  ,       Rio,        ,           BR,     12345,      ,   21203asf124...

(csv files are separated into country folders, so that will be added in afterwards)
"""

#///////////////////////////////////////////////////////////////////////////////////////
#   STEP 1
#   Owner: Archi & Ian
#   description: takes root data location and processes data into dictionary
#
#   Parameters
##      FILE_NAME:  root file location
##      return:     returns list of dictionaries with keys matching opencage
##      eg output:  [{'houseNumber': '3a', 'road': 'Main St.', 'neighborhood': '', 'city': 'Toronto', 'county':...]
def read_csv(root_location):
    pass
    

#   Owner: Archi & Ian
#   description: basically takes the dict and flips it around so that the words point to their tag rather than vice verca.
#   Adding in because Archi and Ian got worried about runtime
#   return: dict mapping entity to its relating tag.
#   eg output:  [{'3a': 'houseNumber', 'Main' : 'road', 'St.' : 'road' ....}, {...}, ... ]
def dict_to_hash(csv_dict)
    csv_dict_flipped = {value:key for key, value in csv_dict.items()}
    return csv_dict_flipped
    
    
    
    
#///////////////////////////////////////////////////////////////////////////////////////
#   STEP 2
#   Owner: Archi & Ian
#   description: takes the dict and enters it into OpenCage to generate the overall sentance as would be entered by human
#
#   Parameters
##      csv_dict:   the dictionary created by read_csv()
##      return:     returns list of opencage generated strings
##      eg output:  ["52 Main St., Unit 3a, Toronto, ON, N6C 4E9","9 South St., Rio BR, 12345",...]
def run_open_cage(csv_dict):
    pass
    
#   Owner: Archi & Ian
#   description: literally runs numpy.zip on the lists generated by run_open_cage() and dict_to_hash(), turning them into a (using Java notation, sorry) List<Touple<String, Dict<String, String>>>
#
#   Parameters
##      cage_strings:   output from run_open_cage() containing the list of all the strings
##      word_hash:      output from dict_to_hash(), containing the list of all of the dictionaries for each string
##      return:         returns one list with touples containing the elements from cage_strings and word_hash with matching indicies
##      eg output:  [("52 Main St., Unit 3a, Toronto, ON, N6C 4E9", {'3a': 'houseNumber', 'Main' : 'road', 'St.' : 'road' ....}),
#                    ("9 South St., Rio BR, 12345",{...}),
#                        ...]
def zip_lists(cage_strings,word_hash):
    pass
    
    
    
    
#///////////////////////////////////////////////////////////////////////////////////////
#   STEP 3
#   Owner: Mona & Saira
#   description: takes lists from zip_lists and for each cage string, looks at the hash of word->tag and assigns the correct tag to that word, formatting all into a string to be put into a file
#
#   Parameters
##      zipped_lists:  output from zip_lists(), list of touples with the human string and the map of word->tag
##      return:     returns finished string of all items to be written to file
##      eg output:  "DOCTYPPE -X- -X- -0-\n3a NPP NPP B-Unit\nMain NPP NPP B-Street\nSt. NPP NPP I-Street ... "
import nltk

def NER_tags(address):
    tags ={}
    for part in address:
        tokens = part['token'].split(' ')
        for i in range(len(tokens)):
            if i == 0:
                tags[tokens[i]] = 'B-' + part['label']
        else:
            tags[tokens[i]] = 'I-' + part['label']
    return tags

def tokenize(address):
    tokens = []
    for part in address:
        tokens = tokens + part['token'].split(' ')
    return tokens


###ADD POS tagging formula here ###
def POS_tags(tokens):
    
        
        
        
        tagged=nltk.pos_tag(tokens)
        
        return dict(tagged)

#   Owner: Mona & Saira
#   description: just writes the output of the above to a file
#
#   Parameters
##      FILE_NAME:  root file location
##      return:     n/a
##      eg output:  complete CONLL file

def to_CoNLL(address):
    tokens = tokenize(address)
    tags = NER_tags(address)
    pos =  POS_tags(tokens)
    conll = '-DOCSTART- -X- -X- O \n'
    for i in range(len(tokens)):
        conll = conll + '{} {} {} {} \n'.format(tokens[i], pos[tokens[i]], pos[tokens[i]], tags[tokens[i]])
    return conll

def write_CONLL_file(zipped_lists):
    file = open(OUT_FILE_NAME, 'w+')
    for address in addresses:
        file.write(to_CoNLL(address))
    file.close()



#Just so you all can see the logic

def main():
    csv_dict = read_csv(FILE_NAME)
    word_hash = dict_to_hash(csv_dict)
    cage_strings = run_open_cage(csv_dict)
    zipped_lists = zip_lists(cage_strings, word_hash)
    write_CONLL_file(zipped_lists)
