import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from g4satbench.data.augmented import SRAugmentedDataset
from g4satbench.data.multisat import MultiSATDataset
from g4satbench.data.sr import SRDataset
from g4satbench.data.three_sat import ThreeSatDataset
from g4satbench.models.gnn import GNN


def opts(**kwargs):
    obj = lambda: None
    for k, v in kwargs.items():
        setattr(obj, k, v)
    return obj


def collate_fn(batch):
    return Batch.from_data_list(batch)


def U(a, b):
    return a, b


def SR(n, samples):
    a, b = n
    return SRDataset(nr_gen_instances=samples, min_n=a, max_n=b)


def augmented_SR(n, samples, augmentations):
    a, b = n
    return SRAugmentedDataset(nr_gen_instances=samples, nr_augm_per_instance=augmentations, min_n=a, max_n=b)


def USat(n, m, l, samples):
    min_n, max_n = n
    min_m, max_m = m
    min_l, max_l = l
    return ThreeSatDataset(samples, min_n, max_n, min_m, max_m, min_l, max_l)

def FileDataset(data_dir):
    return MultiSATDataset(data_dir)


def dataloader(dataset, batch_size=256):
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, pin_memory=True)


def facts(ft):
    return {
        "TP": ft.tp,
        "TN": ft.tn,
        "FP": ft.fp,
        "FN": ft.fn,
        "TPR": ft.tpr(),
        "TNR": ft.tnr(),
        "ACC": ft.accuracy(),
        "F1": ft.f1()
    }


def Model(batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    options = opts(
        model="neurosat",
        graph="lcg",
        init_emb="learned",
        dim=128,
        n_iterations=32,
        n_mlp_layers=2,
        activation="relu",
        task="satisfiability",
        label="satisfiability",
        data_fetching="parallel",
        batch_size=batch_size,
        device=device
    )
    return GNN(options).to(device)


if __name__ == '__main__':
    dataset = SR(n=U(4, 10), samples=10_000)
