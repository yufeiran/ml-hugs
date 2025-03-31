import numpy as np

if __name__=='__main__':
    t_iter = 0

    path = '/mnt/data1/yu/data/ClothGS/output/human_cloth/neuman/seattle/human_cloth-dataset.seq=lab/2025-03-27_16-22-50/ckpt/human_params_{t_iter:06d}.pth.npy'

    # load human params
    human_params = np.load(path.format(t_iter=t_iter),allow_pickle=True).item()

    pass