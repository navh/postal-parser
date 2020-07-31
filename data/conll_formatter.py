file = 'training.txt'

formatted = open(r"conll_formatted.txt", "a", encoding="utf8")   

with open(file, encoding='utf8') as f:
    data_list = f.readlines()

for line in data_list:
    
    bparens_pos = line.find('{')
    eparens_pos = line.find('}')
    new_line = line[bparens_pos+1:eparens_pos]
    new_line = new_line.split(',')
    if len(new_line[0]) > len(new_line[1]):
        new_line = new_line[0]
    else:
        new_line = new_line[1]
    new_line = new_line.split(':')
    new_line = new_line[1]
    new_line = new_line.split('\\n\\n')
    formatted.write(str(new_line[0]).lstrip(" '")+'\n\n')
    new_line = new_line[1]
    new_line = new_line.split('\\n')
    for item in new_line:
        formatted.write(str(item).rstrip("'")+'\n')
    
formatted.close()


