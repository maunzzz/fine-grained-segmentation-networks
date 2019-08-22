from torch.utils import data


class Merged(data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

        self.start_ind = []
        self.stop_ind = []

        start = 0
        for dataset in datasets:
            self.start_ind.append(start)
            self.stop_ind.append(start + len(dataset))

            start += len(dataset)

    def __getitem__(self, index):
        dataset_ind = next(i for i in range(len(self.stop_ind)) if self.stop_ind[i] > index)

        return self.datasets[dataset_ind][index - self.start_ind[dataset_ind]]

    def __len__(self):
        return self.stop_ind[-1]
