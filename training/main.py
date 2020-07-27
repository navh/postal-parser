import nltk
import re
import random
import csv
from google.cloud import storage

# Parameters on robustness of training data
SHUFFLE, DELETE, DUPLICATE = 0.2, 0.2, 0.2
# Training/testing ratio
PERCENT_TRAINING = 0.8
# Mapping of file attribute titles to libpostal fields
HEADER = {'LAT': '',
          'LON': '',
          'NUMBER': 'house_number',
          'STREET': 'road',
          'street': 'road',
          'UNIT': 'unit',
          'CITY': 'city',
          'DISTRICT': '',
          'REGION': 'state',
          'POSTCODE': 'postcode',
          'ID': '',
          'HASH': '',
          'house_number': 'house_number',
          'road': 'road',
          'unit': 'unit',
          'city': 'city',
          'state': 'state',
          'postcode': 'postcode',
          'country': 'country',
          'COUNTRY': 'country'
          }

nltk.download('averaged_perceptron_tagger')

class Address:
    # Representation of a structured address for development of training data

    MAX_NUM_TAGS = 20

    def __init__(self, dictionary, probability_shuffle=0.0, probability_delete=0.0, probability_duplicate=0.0):
        # Description: Builds Address object from a list of dictionaries
        #   Parameter: dictionary
        #       takes in a list of dictionaries of the form
        #       [{'label': ____, 'value': ____},...]
        #   Parameter: order
        #       A tuple or list of lib postal labels (as strings)
        #       in the order expected for the unstructured address
        self.address_dict = dictionary
        self._set_order()
        if random.random() < probability_shuffle:
            self._randomize_order()
        self._duplicate_tags(probability_duplicate)
        self._delete_tags(probability_delete)
        self.ordered = False

    def __str__(self):
        # Description: Converts the class to a string, checks if ordered before returning
        if not self.ordered:
            self.order_address()
        accum_string = ''
        for value in self.address_dict:
            accum_string += value['value']
            accum_string += ' '
        return accum_string.strip()

    def order_address(self):
        # Description: Sorts csv_dict to create a list of dictionaries
        #   such that they are in the same order they would be in an
        #   address string written by a human.  Uses the order stored in class
        address_list = []
        for i in range(len(self.order)):
            counter = 0
            while counter < len(self.address_dict):
                if self.address_dict[counter]['label'].lower() == self.order[i]:
                    address_list.append(self.address_dict[counter])
                    break
                else:
                    counter += 1
        self.address_dict = address_list
        self.ordered = True

    def _set_order(self):
        r = random.randint(0, 2)
        if r == 0:
            new_order = ['house_number', 'road', 'city', 'city_district',
                         'state_district', 'state', 'postcode', 'house', 'level', 'unit', 'po_box',
                         'country']
        elif r == 1:
            new_order = ['house', 'house_number', 'po_box', 'road', 'city',
                         'city_district', 'state_district', 'state', 'postcode', 'level', 'unit',
                         'country']
        else:
            new_order = ['house', 'level', 'unit', 'po_box', 'house_number',
                         'road', 'city', 'city_district',
                         'state_district', 'state', 'postcode', 'country']
        self.ordered = False
        self.order = new_order

    def _randomize_order(self):
        random.shuffle(self.order)
        self.ordered = False

    def _delete_tags(self, _delete_probability):
        while random.random() < _delete_probability and len(self.order) > 1:
            del (self.order[random.randint(0, len(self.order) - 1)])
        self.ordered = False

    def _duplicate_tags(self, _duplicate_probability):
        while random.random() < _duplicate_probability and len(self.order) < self.MAX_NUM_TAGS:
            item_to_be_duplicated = self.order[random.randint(0, len(self.order) - 1)]
            self.order.insert(random.randint(0, len(self.order)), item_to_be_duplicated)
        self.ordered = False

    def to_conll(self):
        # Description: Takes the address in the order stored and develops the CONLL
        #   representation of the address
        tokens = self._tokenize()
        tags = self._ner_tags()
        pos = self._pos_tags(tokens)
        conll = ''
        for i in range(len(tokens)):
            token_val = tokens[i]
            pos_val = pos[i][1]
            tag_val = tags[token_val]
            conll = conll + '{} {} {} {} \n'.format(token_val, pos_val, pos_val, tag_val)
        return conll

    def _ner_tags(self):
        tags = {}
        for part in self.address_dict:
            value = re.split('[ _]', part['value'])
            tokens = []
            tokens = tokens + [word for word in value if word]
            for i in range(len(tokens)):
                if i == 0:
                    tags[tokens[i]] = 'B-' + part['label']
                else:
                    tags[tokens[i]] = 'I-' + part['label']
        return tags

    def _tokenize(self):
        tokens = []
        for part in self.address_dict:
            value = re.split('[ _]', part['value'])
            tokens = tokens + [word for word in value if word]
        return tokens

    def _pos_tags(self, tokens):
        tagged = nltk.pos_tag(tokens)
        return tagged

# Entry function for Google cloud storage upload
def handle_file(event, context):
    try:
        client = storage.Client()
        bkt = client.get_bucket(event['bucket'])
        blob = bkt.get_blob(event['name'])

        print('opening file')

        with open('/tmp/tempfile', 'wb') as file_obj:
            client.download_blob_to_file(blob, file_obj)

        with open('/tmp/tempfile', 'r', newline='') as file_obj:
            # Dealing with a .csv file from OpenAdresses
            if event['name'].endswith('.csv'):
                out_list = read_csv(file_obj, ',')
            # Dealing with a .txt file from OpenCage
            elif event['name'].endswith('inputdict.txt'):
                out_list = [read_txt_dict(list(line)[0]['components']) for line in file_obj]
            # Some other file upload that we don't care about
            else:
                return
            [address.order_address() for address in out_list]
            out = [address.to_conll() for address in out_list]

        trainfile = '/tmp/trainfile'
        testfile = '/tmp/testfile'

        train_file = open(trainfile, 'w+')
        test_file = open(testfile, 'w+')

        conll = "-DOCSTART- -X- -X- O\n\n"
        for address in out:
            if random.random() < PERCENT_TRAINING:
                train_file.write(conll)
                train_file.write(address)
                train_file.write("\n")
            else:
                test_file.write(conll)
                test_file.write(address)
                test_file.write("\n")
        train_file.close()
        test_file.close()

        # Copy files to bucket
        train_blob = bkt.blob('conll-data/temptrain')
        test_blob = bkt.blob('conll-data/temptest')

        train_blob.upload_from_filename(trainfile)
        test_blob.upload_from_filename(testfile)
    except Exception as e:
        print(e)

def read_csv(file_obj, delimiter, parent_info=[]):
    # Opens a .csv file at location file_location and converts each line into a list of Address objects
    out_list = []
    reader = csv.reader(file_obj, delimiter=delimiter)
    headers = next(reader)
    for row in reader:
        line = []
        for i in range(len(row)):
            label = HEADER[headers[i]]
            value = row[i]
            if value and label:
                line.append({'label': label, 'value': value})
        for parent in parent_info:
            line.append(parent)
        out_list.append(
            Address(line, probability_shuffle=SHUFFLE, probability_delete=DELETE, probability_duplicate=DUPLICATE))
    return out_list

# Takes OpenCage output and transforms into a single Address class
def read_txt_dict(line):
    add_list = []
    for field in HEADER:
        if line and field in line:
            add_list.append({'label': HEADER[field], 'value': str(line[field])})

    add = Address(add_list, probability_shuffle=SHUFFLE, probability_delete=DELETE, probability_duplicate=DUPLICATE)
    return add
