import hashlib
input_file = 'crackstation.txt'
output_file = 'crackstation_hash.txt'

counter = 0
with open(input_file) as f:
    for line in f:
        with open(output_file, "a") as myfile:
            myfile.write(hashlib.sha256(line).hexdigest())
            myfile.write('\n')
