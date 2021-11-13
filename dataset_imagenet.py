import os

def prepare(root):
    D = {}
    with open('{:s}/wnids.txt'.format(root)) as f:
        for index, line in enumerate(f):
            D[line.strip()] = index
    for categoryId in D:
        os.system('mv {:s}/train/{:s} {:s}/train/{:d}'.format(root, categoryId, root, D[categoryId]))
        os.system('mkdir -p {:s}/val/{:d}'.format(root, D[categoryId]))
    with open('{:s}/val/val_annotations.txt'.format(root)) as f:
        for line in f:
            imgFileName, categoryId, _ = line.strip().split('\t', 2)
            if categoryId in D:
                os.system('mv {:s}/val/images/{:s} {:s}/val/{:d}'.format(root, imgFileName, root, D[categoryId]))
    os.system('rmdir {:s}/val/images'.format(root))
