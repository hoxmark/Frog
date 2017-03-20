import hashlib
input_file = '10_million_password_list_top_1000000.txt'
output_file = 'passwords_one_line.txt'
output_file_lengths = 'password_lengths.txt'

counter = 0
with open(input_file) as f:
    for line in f:
        with open(output_file, 'a') as password_file:
            password_file.write(line.rstrip())

            with open(output_file_lengths, "a") as length_file:
                length_file.write("{}\n".format(len(line.rstrip())))
