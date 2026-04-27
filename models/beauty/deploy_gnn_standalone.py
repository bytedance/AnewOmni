import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit import RDLogger
from torch.utils.data import DataLoader, Dataset
from torch_scatter import scatter_mean, scatter_sum


RDLogger.DisableLog('rdApp.*')

ATOM_TYPES = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
BOND_TYPES = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, act_fn=nn.SiLU(), end_with_act=False, dropout=0.0):
        super().__init__()
        assert n_layers >= 2
        self.input_linear = nn.Linear(input_size, hidden_size)
        medium_layers = [act_fn]
        for _ in range(n_layers):
            medium_layers.append(nn.Linear(hidden_size, hidden_size))
            medium_layers.append(act_fn)
            medium_layers.append(nn.Dropout(dropout))
        self.medium_layers = nn.Sequential(*medium_layers)
        if end_with_act:
            self.output_linear = nn.Sequential(nn.Linear(hidden_size, output_size), act_fn)
        else:
            self.output_linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_linear(x)
        x = self.medium_layers(x)
        x = self.output_linear(x)
        return x


class GINEConv(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, edge_size, n_layers=3, eps=0):
        super().__init__()
        self.n_layers = n_layers
        self.eps = eps
        self.linear_input = nn.Linear(input_size, hidden_size)
        self.linear_edge = nn.Linear(edge_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, out_size)
        self.mlps = nn.ModuleList(
            [
                MLP(hidden_size, hidden_size, hidden_size, n_layers=2, act_fn=nn.ReLU(), end_with_act=True)
                for _ in range(n_layers)
            ]
        )

    def forward(self, h, edge_index, edge_attr):
        src, dst = edge_index
        h = self.linear_input(h)
        edge_attr = self.linear_edge(edge_attr)
        for i in range(self.n_layers):
            msg = F.relu(h[dst] + edge_attr)
            aggr = scatter_sum(msg, src, dim=0, dim_size=h.shape[0])
            h = self.mlps[i]((1 + self.eps) * h + aggr)
        return self.linear_out(h)


class GNNModel(nn.Module):
    def __init__(self, in_node, in_edge, hidden_size=128, out_dim=1, n_conv=3, readout='mean', dropout=0.1):
        super().__init__()
        self.readout = readout
        self.conv = GINEConv(in_node, hidden_size, hidden_size, in_edge, n_layers=n_conv)
        self.head = MLP(hidden_size, hidden_size, out_dim, n_layers=3, dropout=dropout)

    def encode(self, x, edge_index, edge_attr, batch_index, batch_size):
        h = self.conv(x, edge_index, edge_attr)
        if h.numel() == 0:
            return h.new_zeros((batch_size, h.shape[-1]))
        if self.readout == 'sum':
            return scatter_sum(h, batch_index, dim=0, dim_size=batch_size)
        return scatter_mean(h, batch_index, dim=0, dim_size=batch_size)

    def forward(self, x, edge_index, edge_attr, batch_index, batch_size):
        g = self.encode(x, edge_index, edge_attr, batch_index, batch_size)
        return self.head(g)


def mol_from_smiles(smiles):
    return Chem.MolFromSmiles(smiles)


def atom_feature(atom):
    z = atom.GetAtomicNum()
    onehot = [1.0 if z == t else 0.0 for t in ATOM_TYPES] + [1.0 if z not in ATOM_TYPES else 0.0]
    degree = float(atom.GetDegree())
    charge = float(atom.GetFormalCharge())
    aromatic = 1.0 if atom.GetIsAromatic() else 0.0
    return np.array(onehot + [degree, charge, aromatic], dtype=np.float32)


def bond_feature(bond):
    bt = bond.GetBondType()
    onehot = [1.0 if bt == t else 0.0 for t in BOND_TYPES]
    conj = 1.0 if bond.GetIsConjugated() else 0.0
    return np.array(onehot + [conj], dtype=np.float32)


def graph_from_mol(mol):
    x = np.stack([atom_feature(a) for a in mol.GetAtoms()], axis=0)
    ei_src = []
    ei_dst = []
    eattr = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        f = bond_feature(b)
        ei_src.extend([i, j])
        ei_dst.extend([j, i])
        eattr.extend([f, f])
    if len(eattr) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, len(BOND_TYPES) + 1), dtype=torch.float32)
    else:
        edge_index = torch.tensor([ei_src, ei_dst], dtype=torch.long)
        edge_attr = torch.tensor(np.stack(eattr, axis=0), dtype=torch.float32)
    x = torch.tensor(x, dtype=torch.float32)
    return x, edge_index, edge_attr


class SmilesDataset(Dataset):
    def __init__(self, smiles_list):
        self.smiles = list(smiles_list)
        self.graphs = []
        for smi in self.smiles:
            mol = mol_from_smiles(smi)
            if mol is None:
                raise ValueError(f'无效SMILES: {smi}')
            self.graphs.append(graph_from_mol(mol))

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        x, edge_index, edge_attr = self.graphs[idx]
        return {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr, 'smiles': self.smiles[idx]}


def collate_graph(batch):
    xs = []
    edge_indices = []
    edge_attrs = []
    batch_index = []
    smiles = []
    offset = 0
    for graph_id, item in enumerate(batch):
        x = item['x']
        edge_index = item['edge_index']
        edge_attr = item['edge_attr']
        xs.append(x)
        if edge_index.numel() > 0:
            edge_indices.append(edge_index + offset)
            edge_attrs.append(edge_attr)
        if x.shape[0] > 0:
            batch_index.append(torch.full((x.shape[0],), graph_id, dtype=torch.long))
        smiles.append(item['smiles'])
        offset += x.shape[0]
    x = torch.cat(xs, dim=0) if xs else torch.zeros((0, len(ATOM_TYPES) + 1 + 3), dtype=torch.float32)
    if edge_indices:
        edge_index = torch.cat(edge_indices, dim=1)
        edge_attr = torch.cat(edge_attrs, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, len(BOND_TYPES) + 1), dtype=torch.float32)
    batch_index = torch.cat(batch_index, dim=0) if batch_index else torch.zeros((0,), dtype=torch.long)
    return {
        'x': x,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'batch_index': batch_index,
        'smiles': smiles,
        'batch_size': len(batch),
    }


def build_model_from_ckpt(ckpt):
    args = ckpt['args']
    if args['model_type'] != 'gnn':
        raise ValueError(f'Unsupported model type: {args["model_type"]}')
    task_type = args['task_type']
    out_dim = 1 if task_type == 'regression' else int(args['num_classes'])
    model = GNNModel(
        in_node=len(ATOM_TYPES) + 1 + 3,
        in_edge=len(BOND_TYPES) + 1,
        hidden_size=int(args['gnn_hidden']),
        out_dim=out_dim,
        n_conv=int(args['gnn_layers']),
        readout=str(args.get('readout', 'mean')),
        dropout=float(args['dropout']),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    return model


def parse_ckpt_paths(values):
    paths = []
    for item in values:
        for path in str(item).split(','):
            path = path.strip()
            if path:
                paths.append(path)
    if not paths:
        raise ValueError(f'No valid checkpoints detected')
    return paths


def load_predictors(ckpt_paths, device):
    predictors = []
    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model = build_model_from_ckpt(ckpt).to(device).eval()
        args = ckpt['args']
        task_type = args['task_type']
        predictor = {
            'path': ckpt_path,
            'task_type': task_type,
            'target_mean': float(ckpt['target_mean']),
            'target_std': float(ckpt['target_std']),
            'model': model,
        }
        if task_type == 'classification':
            bin_edges = np.asarray(ckpt['bin_edges_norm'], dtype=np.float64)
            predictor['centers'] = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        predictors.append(predictor)
    return predictors


def predict_batch(batch, predictors, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    per_model_pred_norm = []
    per_model_pred_raw = []
    with torch.no_grad():
        for predictor in predictors:
            logits = predictor['model'](
                batch['x'],
                batch['edge_index'],
                batch['edge_attr'],
                batch['batch_index'],
                batch['batch_size'],
            )
            if predictor['task_type'] == 'regression':
                pred_norm = logits.squeeze(-1).detach().cpu().numpy()
            else:
                prob = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                pred_norm = (prob * predictor['centers'].reshape(1, -1)).sum(axis=-1)
            pred_raw = pred_norm * predictor['target_std'] + predictor['target_mean']
            per_model_pred_norm.append(pred_norm)
            per_model_pred_raw.append(pred_raw)
    ensemble_pred_norm = np.mean(np.stack(per_model_pred_norm, axis=0), axis=0)
    ensemble_pred_raw = np.mean(np.stack(per_model_pred_raw, axis=0), axis=0)
    rows = []
    for i, smi in enumerate(batch['smiles']):
        row = {
            'SMILES': smi,
            'prediction_norm': float(ensemble_pred_norm[i]),
            'prediction_raw': float(ensemble_pred_raw[i]),
        }
        rows.append(row)
    return rows


def build_parser():
    default_ckpt = os.path.join(os.path.dirname(__file__), 'best.pt')
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', nargs='+', default=[default_ckpt])
    parser.add_argument('--smiles', nargs='+', default=None)
    parser.add_argument('--csv', default=None)
    parser.add_argument('--csv-smiles-col', default='SMILES')
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', default=None)
    return parser


def main():
    args = build_parser().parse_args()
    smiles_list = []
    if args.smiles:
        smiles_list.extend(args.smiles)
    if args.csv:
        df = pd.read_csv(args.csv)
        if args.csv_smiles_col not in df.columns:
            raise ValueError(f'No column named {args.csv_smiles_col} in the provided csv')
        smiles_list.extend(df[args.csv_smiles_col].astype(str).tolist())
    if not smiles_list:
        raise ValueError('No smiles detected')
    ckpt_paths = parse_ckpt_paths(args.ckpt)
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    predictors = load_predictors(ckpt_paths, device)
    dataset = SmilesDataset(smiles_list)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_graph)
    rows = []
    for batch in loader:
        rows.extend(predict_batch(batch, predictors, device))
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    main()
