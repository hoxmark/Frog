base_dir = "/datadrive/cracklist/testing"
input_file = '{}/testing.txt'.format(base_dir)
output_file = '{}/passwords_one_line.txt'.format(base_dir)
output_file_lengths = '{}/password_lengths.txt'.format(base_dir)

with open(input_file) as f:
	with open(output_file_lengths, "a") as length_file:
		with open(output_file, 'a') as password_file:
		    for line in f:
			    password_file.write(line.rstrip())
			    length_file.write("{}\n".format(len(line.rstrip())))