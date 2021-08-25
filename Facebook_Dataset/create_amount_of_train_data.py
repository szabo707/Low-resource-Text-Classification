import math

for PERCENTAGE in range(10,100,10):
    with open("czech_facebook_train.txt", "r") as train_data:
        train_lines = train_data.readlines()
        number_of_lines = len(train_lines)
        number_of_output_lines = math.ceil((number_of_lines / 100) * PERCENTAGE)
        output_file = "czech_facebook_" + str(PERCENTAGE) + "_train.txt"

        with open(output_file, "w") as writer:
            for i in range(number_of_output_lines):
                writer.write(train_lines[i])


    print(number_of_lines)
    print(sum(1 for line in open(output_file)))
    print()



