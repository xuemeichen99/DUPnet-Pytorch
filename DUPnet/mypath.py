class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'LR2022':
            return './LRDataset/LR2022/'

        else:
            print('Dat''aset {} not available.'.format(dataset))
            raise NotImplementedError
