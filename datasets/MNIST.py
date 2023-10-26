import torchvision
import numpy as np
from datasets.data_transform import mnist_transform


class MNIST(torchvision.datasets.MNIST):
    def __init__(self, root, train, download, data_augment=False, forget_type=None,
                 forget_num=None, only_forget=False, random_exper=False):
        transform = mnist_transform(data_augment)
        super().__init__(root, train=train, download=download, transform=transform)
        if random_exper:
            random_id = np.random.choice(list(range(len(self.targets))), forget_num, replace=False)
            num_class = self.num_class()
            random_target = np.random.randint(num_class, size=len(random_id))
            for i in range(forget_num):
                self.targets[random_id[i]] = random_target[i]
            self.random_id = random_id
            print("random index is {}, random target is {}".format(random_id[:5], random_target[:5]))
        if forget_type is not None:
            remove_id = []
            if forget_type == 'class':
                for i in range(len(self.targets)):
                    if self.targets[i] == forget_num:
                        remove_id.append(i)
            elif forget_type == 'random':
                remove_id = random_id
            else:
                print("error, unknown forget type!")
            if not only_forget:
                print("forget {} {} is prepared , remain data {}/{}".
                      format(forget_type, forget_num, len(self.targets) - len(remove_id), len(self.targets)))
                self.data = np.delete(self.data, remove_id, axis=0)
                self.targets = np.delete(np.array(self.targets), remove_id).tolist()
            else:
                print("only forget {} {} is prepared , remain data {}/{}".
                      format(forget_type, forget_num, len(remove_id), len(self.targets)))
                self.data = self.data[remove_id]
                self.targets = np.array(self.targets)[remove_id].tolist()

    def num_class(self):
        return len(np.unique(np.array(self.targets)))