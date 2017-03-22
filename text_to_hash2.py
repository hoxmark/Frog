import hashlib
input_file = '/datadrive/cracklist/crackstation.txt'
output_file = '/datadrive/cracklist/passwords_one_line.txt'
output_file_lengths = '/datadrive/cracklist/password_lengths.txt'

counter = 0
with open(input_file) as f:
	with open(output_file_lengths, "a") as length_file:
		with open(output_file, 'a') as password_file:
		    for line in f:        
			    password_file.write(line.rstrip())            
			    length_file.write("{}\n".format(len(line.rstrip())))
