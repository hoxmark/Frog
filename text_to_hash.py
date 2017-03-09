import hashlib
input_file = '10_million_password_list_top_1000000.txt'
output_file = 'crackstation_hash.txt'

counter = 0
with open(input_file) as f:
    for line in f:
        with open(output_file, "a") as myfile:
            myfile.write(hashlib.sha256(line).hexdigest())
            myfile.write('\n')
