import pandas


class Dataset:
    def __init__(self, pathordata, filepath=None, filetype="csv", data=None):
        if pathordata == 0:
            if filetype == "csv":
                self.data = pandas.read_csv(filepath)
            else:
                print("File type not supported")
        else:
            self.data = pandas.DataFrame(data)
        self.batches=[]

    def make_batch(self, batch_size, random_seed):
        if batch_size<=self.data.shape[0]:
            random_rows = self.data.sample(n=batch_size, random_state=random_seed)
            self.batches.append(Dataset(1,data=random_rows))
        else:
            print('batch_size exceeds datas')
    
    def __len__(self):
        return self.data.shape[0]
