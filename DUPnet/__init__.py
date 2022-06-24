import LRDataset
from torch.utils.data import DataLoader


def make_data_loader(args, **kwargs):
    # 添加Abyssinian
    if args.dataset == 'LR2022':
        train_set = LRDataset.Segmentation(args, split='train')
        val_set = LRDataset.Segmentation(args, split='val')
        '''

        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])
         '''
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, drop_last= True,batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, drop_last=  True,batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
    else:
        raise NotImplementedError