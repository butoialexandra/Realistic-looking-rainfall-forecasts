import numpy as np
import torch

from tqdm.auto import tqdm
import dataset


def test_load_observations():
    obs = dataset.load_observations()
    assert len(obs) > 0

def test_train_test_split():
    ds = dataset.Dataset()
    image_ids = ds.get_image_ids()
    assert type(image_ids) == list
    # 730 per month * m months = 39432 obs
    assert len(image_ids) > 0

    train_ids, test_ids = ds.train_test_split_ids()
    assert type(train_ids) == list
    assert type(test_ids) == list
    assert len(train_ids) > len(test_ids)

def test_train_iterator():
    training_params = {"batch_size": 8, "shuffle": True, "num_workers": 0}
    train_ds = dataset.Dataset(device='cpu')
    train_idx, test_idx = train_ds.train_test_split_ids()
    train_ds.select_indices(train_idx)
    assert len(train_ds) > 0
    training_generator = torch.utils.data.DataLoader(train_ds, **training_params)
    for i, (pred_imgs, real_imgs) in enumerate(training_generator):
        assert pred_imgs.shape[0] == real_imgs.shape[0] == 8
        # test if the examples are valid
        assert not torch.any(pred_imgs.isnan())
        assert not torch.any(real_imgs.isnan())
        if i > 100:
            break

def find_nan_images():
    training_params = {"batch_size": 8, "shuffle": True, "num_workers": 0}
    train_ds = dataset.Dataset(device='cpu')
    train_ds.select_all()
    assert len(train_ds) > 0
    # training_generator = torch.utils.data.DataLoader(train_ds, **training_params)
    for i, (x, y) in tqdm(enumerate(train_ds), total=len(train_ds)):
        pass

def test_reftimes():
    ds = dataset.Dataset(device='cpu')
    reftimes = ds.cosmo['reftime']
    dt = reftimes[0]
    for item in reftimes[1:]:
        rt = item.values
        if (rt != dt + np.timedelta64(12, 'h')) and \
           (rt != dt + np.timedelta64(24, 'h')) and \
           (rt != dt + np.timedelta64(36, 'h')) and \
           (rt != dt + np.timedelta64(48, 'h')) and \
           (rt != dt + np.timedelta64(60, 'h')):  # max. 2.5 days
            raise Exception(f"{dt} + 1d != {rt}")
        dt = rt

if __name__ == "__main__":
    # foo = dataset.load_predictions()
    # print(foo['chx'].min())

    # bar = dataset.load_observations()
    # print(bar['201805']['chx'].min())
    find_nan_images()
    # test_reftimes()

    # training_params = {"batch_size": 8, "shuffle": True, "num_workers": 0}
    # train_ds = dataset.Dataset(device='cpu')
    # print(train_ds.compute_nearest_neighbors())

    # train_idx, test_idx = train_ds.train_test_split_ids()
    # train_ds.select_indices(train_idx)
    # print(len(train_ds), "training datapoints")
    # training_generator = torch.utils.data.DataLoader(train_ds, **training_params)
    # for i, (pred_imgs, real_imgs) in enumerate(training_generator):
    #     assert pred_imgs.shape[0] == real_imgs.shape[0] == 8
    #     break
