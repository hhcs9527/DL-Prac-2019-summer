class Data:
    def __init__(self, file_path):
        self.target = file_path

    
    def find_valid(self, line):
        sepline =  []
        valid_line = []
        for i in range(len(line)):
            if (line[i] not in sepline or line[i].isdigit()) and i < 4:
                sepline.append(line[i])
                valid_line.append(line[i])
        return valid_line

    # Brute force
    def read_train_file(self):
        load_data = []
        with open(self.target, encoding='utf-8') as f:
            for line in iter(f):
                line = line.split( )
                valid_line = self.find_valid(line)

                # valid tense change
                for i in range(len(valid_line)):
                    #for j in range(len(valid_line)):
                        #change_tense = [line[i], line[j], j]
                        change_tense = [line[i], line[i], i]
                        load_data.append(change_tense)
        return load_data


    def read_test_file(self):
        load_data = []
        with open(self.target, encoding='utf-8') as f:
                for line in iter(f):
                    line = line.split( )      
                    load_data.append(line)      
        return load_data


    def seperate_tense(self):
        # Reclassify the data
        present =[]
        third_person = []
        present_progressive =[]
        simple_past =[]
        train_set = self.read_train_file()

        for i in range(len(train_set)):
            if train_set[i][2] == 0:
                present.append(train_set[i])
            elif train_set[i][2] == 1:
                third_person.append(train_set[i])
            elif train_set[i][2] == 2:
                present_progressive.append(train_set[i])
            else:
                simple_past.append(train_set[i])
        
        return  present, third_person, present_progressive, simple_past

if __name__ == '__main__':
    a = Data('./lab3/train.txt')
    present, third_person, present_progressive, simple_past = a.seperate_tense()


