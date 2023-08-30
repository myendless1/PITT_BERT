import torch
import torch.nn as nn
import yaml
import h5py
from torch.utils.data import ConcatDataset
from transformers import BertModel, BertConfig, AutoModelForMaskedLM

from utils import TransformerOperatorDatasetBert, TransformerOperatorDataset
from tqdm import tqdm
import time
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import shutil

from models.pitt import PhysicsInformedTokenTransformerBert
from models.pitt import StandardPhysicsInformedTokenTransformer

from models.oformer import Encoder1D, STDecoder1D, OFormer1D, PointWiseDecoder1D
from models.fno import FNO1d
from models.deeponet import DeepONet1D

device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')


def custom_collate(batch):
    x0 = torch.empty((len(batch), batch[0][0].shape[0]))
    y = torch.empty((len(batch), batch[0][1].shape[0], 1))
    grid = torch.empty((len(batch), batch[0][2].shape[0]))
    tokens = torch.empty((len(batch), batch[0][3].shape[0]))
    forcing = []
    time = torch.empty(len(batch))
    for idx, b in enumerate(batch):
        x0[idx] = b[0]
        y[idx] = b[1]
        grid[idx] = b[2]
        tokens[idx] = b[3]
        forcing.append(b[4])
        time[idx] = b[5]
    return x0, y, grid, tokens, forcing, time


def progress_plots(ep, y_train_true, y_train_pred, y_val_true, y_val_pred, path="progress_plots", seed=None):
    ncols = 8
    fig, ax = plt.subplots(ncols=ncols, nrows=2, figsize=(5 * ncols, 14))
    for i in range(ncols):
        ax[0][i].plot(y_train_true[i].reshape(100, ).detach().cpu())
        ax[0][i].plot(y_train_pred[i].reshape(100, ).detach().cpu())
        ax[1][i].plot(y_val_true[i].reshape(100, ).detach().cpu())
        ax[1][i].plot(y_val_pred[i].reshape(100, ).detach().cpu())

    fname = str(ep)
    while (len(fname) < 8):
        fname = '0' + fname
    if (seed is not None):
        plt.savefig("./{}/{}_{}.png".format(path, seed, fname))
    else:
        plt.savefig("./{}/{}.png".format(path, fname))
    plt.close()


def val_plots(ep, val_loader, preds, path="progress_plots", seed=None):
    im_num = 0
    for vals in val_loader:
        for idx, v in tqdm(enumerate(vals[1])):

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(v.reshape(100, ).detach().cpu())
            ax.plot(preds[0][idx].detach().cpu())
            fname = str(im_num)
            while (len(fname) < 8):
                fname = '0' + fname
            ax.set_title(fname)
            plt.savefig("./val_2/{}_{}.png".format(seed, fname))
            plt.close()

            im_num += 1


def evaluate(test_loader, transformer, loss_fn):
    # src_mask = generate_square_subsequent_mask(640).cuda()
    with torch.no_grad():
        transformer.eval()
        test_loss = 0
        for bn, (x0, y, grid, tokens, t) in enumerate(test_loader):
            # for bn, (x, y, grid, tokens, x0) in enumerate(test_loader):
            # Forward pass: compute predictions by passing the input sequence
            # through the transformer.
            # y_pred = transformer(grid, tokens.cuda(), x0)
            y_pred = transformer(grid.to(device=device), tokens.to(device=device), x0.to(device=device),
                                 t.to(device=device), device=device)
            # y_pred = transformer(grid.to(device=device), x0.to(device=device), x0.to(device=device), t.to(device=device))
            # y_pred = transformer(grid.cuda(), tokens.cuda(), x0.cuda())#, src_mask)#[:,0,:]
            y = y[..., 0].to(device=device)

            # Compute the loss.
            test_loss += loss_fn(y_pred, y).item()

    return test_loss / (bn + 1)


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def get_neural_operator(model_name, config):
    if (model_name == "fno"):
        neural_operator = FNO1d(config['num_channels'], config['modes'], config['width'], config['initial_step'],
                                config['dropout'])
    elif (model_name == "deeponet"):
        neural_operator = DeepONet1D(layer_sizes_branch=config['branch_net'], layer_sizes_trunk=config['trunk_net'],
                                     activation=config['activation'],
                                     kernel_initializer=config['kernel_initializer'])
    elif (model_name == "oformer"):
        encoder = Encoder1D(input_channels=config['input_channels'], in_emb_dim=config['in_emb_dim'],
                            out_seq_emb_dim=config['out_seq_emb_dim'], depth=config['depth'], dropout=config['dropout'],
                            res=config['enc_res'])
        decoder = PointWiseDecoder1D(latent_channels=config['latent_channels'], out_channels=config['out_channels'],
                                     decoding_depth=config['decoding_depth'], scale=config['scale'],
                                     res=config['dec_res'])
        neural_operator = OFormer1D(encoder, decoder)

    neural_operator.to(device)
    return neural_operator


def get_transformer(model_name, neural_operator, config):
    global transformer
    model_config = BertConfig.from_pretrained('models/BERT/bert-base-uncased')
    model_path = 'models/BERT/bert-base-uncased'
    # 修改配置
    model_config.output_hidden_states = True
    model_config.output_attentions = False
    # 通过配置和路径导入模型
    bert_model = BertModel.from_pretrained(model_path, config=model_config, ignore_mismatched_sizes=True)
    # bert_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    for name, parameter in bert_model.named_parameters():
        parameter.requires_grad = False
    if config['embedding'] == 'standard':
        print("\nUSING STANDARD EMBEDDING")
        transformer = StandardPhysicsInformedTokenTransformer(500, config['hidden'], config['layers'], config['heads'],
                                                              config['num_x'], dropout=config['dropout'],
                                                              bert_model=bert_model,
                                                              neural_operator=neural_operator).to(device=device)
    elif (config['embedding'] == 'novel'):
        print("\nUSING NOVEL EMBEDDING")
        transformer = PhysicsInformedTokenTransformerBert(500, config['hidden'], config['layers'], config['heads'],
                                                          config['num_x'], dropout=config['dropout'],
                                                          bert_model=bert_model,
                                                          neural_operator=neural_operator).to(device=device)
    return transformer


# TODO abandon this method
def get_transformer_tuning(model_name, neural_operator, config):
    global transformer
    model_config = BertConfig.from_pretrained('models/BERT/bert-base-uncased')
    model_path = 'models/BERT/bert-base-uncased'
    # 修改配置
    model_config.output_hidden_states = True
    model_config.output_attentions = False
    # 通过配置和路径导入模型
    bert_model = BertModel.from_pretrained(model_path, config=model_config, ignore_mismatched_sizes=True)
    # bert_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    for name, parameter in bert_model.named_parameters():
        parameter.requires_grad = True
    if config['embedding'] == 'standard':
        print("\nUSING STANDARD EMBEDDING")
        transformer = StandardPhysicsInformedTokenTransformer(500, config['hidden'], config['layers'], config['heads'],
                                                              config['num_x'], dropout=config['dropout'],
                                                              bert_model=bert_model,
                                                              neural_operator=neural_operator).to(device=device)
    elif (config['embedding'] == 'novel'):
        print("\nUSING NOVEL EMBEDDING")
        transformer = PhysicsInformedTokenTransformerBert(500, config['hidden'], config['layers'], config['heads'],
                                                          config['num_x'], dropout=config['dropout'],
                                                          bert_model=bert_model,
                                                          neural_operator=neural_operator).to(
            device=device)

    transformer.load_state_dict(torch.load(config['fusion_model_path'])['model_state_dict'])
    return transformer


def get_data(f, config):
    train_data = TransformerOperatorDataset(f, config['flnm'],
                                            split="train",
                                            initial_step=config['initial_step'],
                                            reduced_resolution=config['reduced_resolution'],
                                            reduced_resolution_t=config['reduced_resolution_t'],
                                            reduced_batch=config['reduced_batch'],
                                            saved_folder=config['base_path'],
                                            return_text=config['return_text'],
                                            num_t=config['num_t'],
                                            num_x=config['num_x'],
                                            sim_time=config['sim_time'],
                                            num_samples=config['num_samples'],
                                            train_style=config['train_style'],
                                            rollout_length=config['rollout_length'],
                                            seed=config['seed'],
                                            val_ratio=0.2,
                                            )
    train_data.data = train_data.data.to(device)
    train_data.grid = train_data.grid.to(device)
    val_data = TransformerOperatorDataset(f, config['flnm'],
                                          split="val",
                                          initial_step=config['initial_step'],
                                          reduced_resolution=config['reduced_resolution'],
                                          reduced_resolution_t=config['reduced_resolution_t'],
                                          reduced_batch=config['reduced_batch'],
                                          saved_folder=config['base_path'],
                                          return_text=config['return_text'],
                                          num_t=config['num_t'],
                                          num_x=config['num_x'],
                                          sim_time=config['sim_time'],
                                          num_samples=config['num_samples'],
                                          train_style=config['train_style'],
                                          rollout_length=config['rollout_length'],
                                          seed=config['seed'],
                                          val_ratio=0.2,
                                          )
    val_data.data = val_data.data.to(device)
    val_data.grid = val_data.grid.to(device)
    test_data = TransformerOperatorDataset(f, config['flnm'],
                                           split="test",
                                           initial_step=config['initial_step'],
                                           reduced_resolution=config['reduced_resolution'],
                                           reduced_resolution_t=config['reduced_resolution_t'],
                                           reduced_batch=config['reduced_batch'],
                                           saved_folder=config['base_path'],
                                           return_text=config['return_text'],
                                           num_t=config['num_t'],
                                           num_x=config['num_x'],
                                           sim_time=config['sim_time'],
                                           num_samples=config['num_samples'],
                                           train_style=config['train_style'],
                                           rollout_length=config['rollout_length'],
                                           seed=config['seed'],
                                           val_ratio=0.2,
                                           )
    test_data.data = test_data.data.to(device)
    test_data.grid = test_data.grid.to(device)
    return train_data, val_data, test_data


def get_data_bert(f, config):
    train_data = TransformerOperatorDatasetBert(f, config['flnm'],
                                                split="train",
                                                initial_step=config['initial_step'],
                                                reduced_resolution=config['reduced_resolution'],
                                                reduced_resolution_t=config['reduced_resolution_t'],
                                                reduced_batch=config['reduced_batch'],
                                                saved_folder=config['base_path'],
                                                return_text=config['return_text'],
                                                num_t=config['num_t'],
                                                num_x=config['num_x'],
                                                sim_time=config['sim_time'],
                                                num_samples=config['num_samples'],
                                                train_style=config['train_style'],
                                                rollout_length=config['rollout_length'],
                                                seed=config['seed'],
                                                val_ratio=0.2,
                                                )
    train_data.data = train_data.data.to(device)
    train_data.grid = train_data.grid.to(device)
    val_data = TransformerOperatorDatasetBert(f, config['flnm'],
                                              split="val",
                                              initial_step=config['initial_step'],
                                              reduced_resolution=config['reduced_resolution'],
                                              reduced_resolution_t=config['reduced_resolution_t'],
                                              reduced_batch=config['reduced_batch'],
                                              saved_folder=config['base_path'],
                                              return_text=config['return_text'],
                                              num_t=config['num_t'],
                                              num_x=config['num_x'],
                                              sim_time=config['sim_time'],
                                              num_samples=config['num_samples'],
                                              train_style=config['train_style'],
                                              rollout_length=config['rollout_length'],
                                              seed=config['seed'],
                                              val_ratio=0.2,
                                              )
    val_data.data = val_data.data.to(device)
    val_data.grid = val_data.grid.to(device)
    test_data = TransformerOperatorDatasetBert(f, config['flnm'],
                                               split="test",
                                               initial_step=config['initial_step'],
                                               reduced_resolution=config['reduced_resolution'],
                                               reduced_resolution_t=config['reduced_resolution_t'],
                                               reduced_batch=config['reduced_batch'],
                                               saved_folder=config['base_path'],
                                               return_text=config['return_text'],
                                               num_t=config['num_t'],
                                               num_x=config['num_x'],
                                               sim_time=config['sim_time'],
                                               num_samples=config['num_samples'],
                                               train_style=config['train_style'],
                                               rollout_length=config['rollout_length'],
                                               seed=config['seed'],
                                               val_ratio=0.2,
                                               )
    test_data.data = test_data.data.to(device)
    test_data.grid = test_data.grid.to(device)
    return train_data, val_data, test_data


def get_data_loader(f, config):
    train_data, val_data, test_data = get_data(f, config)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config['batch_size'],
                                               num_workers=config['num_workers'], shuffle=True,
                                               generator=torch.Generator(device='cuda'))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config['batch_size'],
                                             num_workers=config['num_workers'], shuffle=False,
                                             generator=torch.Generator(device='cuda'))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config['batch_size'],
                                              num_workers=config['num_workers'], shuffle=False,
                                              generator=torch.Generator(device='cuda'))

    # Check against data leaks
    assert not (bool(set(train_data.data_list) & \
                     set(val_data.data_list)) | \
                bool(set(train_data.data_list) & \
                     set(test_data.data_list)) & \
                bool(set(val_data.data_list) & \
                     set(test_data.data_list)))

    return train_loader, val_loader, test_loader


def get_1d_data(f1, config1, f2, config2, f3, config3):
    train_data1, val_data1, test_data1 = get_data_bert(f1, config1)
    train_data2, val_data2, test_data2 = get_data_bert(f2, config2)
    train_data3, val_data3, test_data3 = get_data_bert(f3, config3)

    train_data = ConcatDataset([
        train_data1,
        train_data2,
        train_data3
    ])
    val_data = ConcatDataset([
        val_data1,
        val_data2,
        val_data3
    ])
    test_data = ConcatDataset([
        test_data1,
        test_data2,
        test_data3
    ])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config1['batch_size'],
                                               num_workers=config1['num_workers'], shuffle=True,
                                               generator=torch.Generator(device='cuda'))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config1['batch_size'],
                                             num_workers=config1['num_workers'], shuffle=False,
                                             generator=torch.Generator(device='cuda'))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config1['batch_size'],
                                              num_workers=config1['num_workers'], shuffle=False,
                                              generator=torch.Generator(device='cuda'))

    # Check against data leaks
    assert not (bool(set(train_data1.data_list) & \
                     set(val_data1.data_list)) | \
                bool(set(train_data1.data_list) & \
                     set(test_data1.data_list)) & \
                bool(set(val_data1.data_list) & \
                     set(test_data1.data_list)))
    # assert not (bool(set(train_data2.data_list) & \
    #                  set(val_data2.data_list)) | \
    #             bool(set(train_data2.data_list) & \
    #                  set(test_data2.data_list)) & \
    #             bool(set(val_data2.data_list) & \
    #                  set(test_data2.data_list)))
    # assert not (bool(set(train_data3.data_list) & \
    #                  set(val_data3.data_list)) | \
    #             bool(set(train_data3.data_list) & \
    #                  set(test_data3.data_list)) & \
    #             bool(set(val_data3.data_list) & \
    #                  set(test_data3.data_list)))

    return train_loader, val_loader, test_loader


def run_training(config, prefix):
    # print(f'Epochs = {config[\'epochs\']}, learning rate = {config[\'learning_rate\']}, scheduler step = {scheduler_step}    , scheduler gamma = {scheduler_gamma}')

    ################################################################
    # load data
    ################################################################

    path = "{}{}_{}_{}".format(config['results_dir'], config['transformer'], config['neural_operator'], prefix)
    f = h5py.File("{}{}".format(config['base_path'], config['data_name']), 'r')
    model_name = config['flnm'] + '_{}'.format(config['transformer']) + "_{}.pt".format(seed)
    model_path = path + "/" + model_name

    print("Filename: {}, Seed: {}\n".format(config['flnm'], config['seed']))

    heat_config = {
        'flnm': "Heat",
        'data_name': 'varied_heat_10000.h5',
        'initial_step': 10,
        'reduced_resolution': 1,
        'reduced_resolution_t': 1,
        'reduced_batch': 1,
        'base_path': './pde_data/',
        'return_text': True,
        'num_t': 100,
        'num_x': 100,
        'sim_time': 1000,
        'num_samples': config['num_samples'],
        'train_style': 'next_step',
        'rollout_length': 1,
        'seed': config['seed'],

        'batch_size': 128,
        'num_workers': 0,

    }
    KdV_config = {
        'flnm': "KdV",
        'data_name': 'varied_kdv_2500.h5',
        'initial_step': 10,
        'reduced_resolution': 1,
        'reduced_resolution_t': 1,
        'reduced_batch': 1,
        'base_path': './pde_data/',
        'return_text': True,
        'num_t': 100,
        'num_x': 100,
        'sim_time': 1000,
        'num_samples': config['num_samples'],
        'train_style': 'next_step',
        'rollout_length': 1,
        'seed': config['seed'],

        'batch_size': 128,
        'num_workers': 0,

    }
    burgers_config = {
        'flnm': "Burgers",
        'data_name': 'varied_burgers_2500.h5',
        'initial_step': 10,
        'reduced_resolution': 1,
        'reduced_resolution_t': 1,
        'reduced_batch': 1,
        'base_path': './pde_data/',
        'return_text': True,
        'num_t': 100,
        'num_x': 100,
        'sim_time': 1000,
        'num_samples': config['num_samples'],
        'train_style': 'next_step',
        'rollout_length': 1,
        'seed': config['seed'],

        'batch_size': 128,
        'num_workers': 0,

    }

    heat_f = h5py.File("{}{}".format(heat_config['base_path'], heat_config['data_name']), 'r')
    KdV_f = h5py.File("{}{}".format(KdV_config['base_path'], KdV_config['data_name']), 'r')
    burgers_f = h5py.File("{}{}".format(burgers_config['base_path'], burgers_config['data_name']), 'r')

    train_loader, val_loader, test_loader = get_1d_data(heat_f, heat_config, KdV_f, KdV_config, burgers_f,
                                                        burgers_config)

    neural_operator = get_neural_operator(config['neural_operator'], config)
    # transformer = get_transformer_tuning(config['transformer'], neural_operator, config)
    transformer = get_transformer(config['transformer'], neural_operator, config)

    total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f'Total parameters = {total_params}')

    # Use Adam as the optimizer.
    print("\nWEIGHT DECAY: {}\n".format(config['weight_decay']))
    optimizer = torch.optim.Adam(transformer.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['learning_rate'],  # div_factor=1e6,
                                                    steps_per_epoch=len(train_loader), epochs=config['epochs'])

    # Use mean squared error as the loss function.
    # loss_fn = nn.MSELoss(reduction='mean')
    loss_fn = nn.L1Loss(reduction='mean')

    # Train the transformer for the specified number of epochs.
    train_losses = []
    val_losses = []
    loss_val_min = np.infty
    lrs = []
    shift = 0
    for epoch in tqdm(range(config['epochs'])):
        # Iterate over the training dataset.
        train_loss = 0
        times = []
        max_val = 0
        # for x, y, grid, tokens in tqdm(train_loader):
        transformer.train()
        for bn, (x0, y, grid, tokens, t) in enumerate(train_loader):

            start = time.time()
            y_pred = transformer(grid.to(device=device), tokens.to(device=device), x0.to(device=device),
                                 t.to(device=device), device=device)
            y = y[..., 0].to(device=device)  # .cuda()

            pred_time = time.time()
            print(f"pred_time={pred_time - start}")

            # Compute the loss.
            loss = loss_fn(y_pred, y)

            loss_time = time.time()
            print(f"loss_time={loss_time - pred_time}")

            # Backward pass: compute gradient of the loss with respect to model
            # parameters.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])

            backward_time = time.time()
            print(f"back_time={backward_time - loss_time}")

            train_loss += loss.item()
            if (bn == 0):
                y_train_true = y.clone()
                y_train_pred = y_pred.clone()

            scheduler.step()

            if (bn % 1000 == 0 and len(train_loader) >= 1000):
                print("Batch: {0}\tloss = {1:.6f}".format(bn, train_loss / (bn + 1)))

        # scheduler.step()

        train_loss /= (bn + 1)
        train_losses.append(train_loss)

        with torch.no_grad():
            transformer.eval()
            val_loss = 0
            all_val_preds = []
            for bn, (x0, y, grid, tokens, t) in enumerate(val_loader):
                # Forward pass: compute predictions by passing the input sequence
                # through the transformer.
                y_pred = transformer(grid.to(device=device), tokens.to(device=device), x0.to(device=device),
                                     t.to(device=device), device=device)
                y = y[..., 0].to(device=device)  # .cuda()
                if bn == 0:
                    y_val_true = y.clone()
                    y_val_pred = y_pred.clone()
                all_val_preds.append(y_pred.detach())

                # Compute the loss.
                val_loss += loss_fn(y_pred, y).item()

            if val_loss < loss_val_min:
                loss_val_min = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': transformer.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_val_min
                }, model_path)

        val_loss /= (bn + 1)
        val_losses.append(val_loss)

        # Print the loss at the end of each epoch.
        if (epoch % config['log_freq'] == 0):
            np.save("./{}/train_l2s_{}.npy".format(path, seed), train_losses)
            np.save("./{}/val_l2s_{}.npy".format(path, seed), val_losses)
            np.save("./{}/lrs_{}.npy".format(path, seed), lrs)
            print(f"Epoch {epoch + 1}: loss = {train_loss:.6f}\t val loss = {val_loss:.6f}")

        if (epoch % config['progress_plot_freq'] == 0 and len(y_train_true) >= 4):
            progress_plots(epoch, y_train_true, y_train_pred, y_val_true, y_val_pred, path, seed=seed)

    try:
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(22, 15))
        im0 = ax[0][0].imshow(transformer.query_matrix.detach().cpu()[0])
        divider = make_axes_locatable(ax[0][0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im0, cax=cax, orientation='vertical')

        im1 = ax[0][1].imshow(transformer.key_matrix.detach().cpu()[0])
        divider = make_axes_locatable(ax[0][1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')

        prod_mat = torch.mul(transformer.query_matrix.detach().cpu()[0], transformer.key_matrix.detach().cpu()[0])
        im2 = ax[0][2].imshow(prod_mat, vmin=prod_mat.mean() - prod_mat.std(), vmax=prod_mat.mean() + prod_mat.std())
        divider = make_axes_locatable(ax[0][2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im2, cax=cax, orientation='vertical')

        im3 = ax[1][0].imshow(transformer.q2_matrix.detach().cpu()[0])
        divider = make_axes_locatable(ax[1][0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im3, cax=cax, orientation='vertical')

        im4 = ax[1][1].imshow(transformer.k2_matrix.detach().cpu()[0])
        divider = make_axes_locatable(ax[1][1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im4, cax=cax, orientation='vertical')

        prod_mat2 = torch.mul(transformer.q2_matrix.detach().cpu()[0], transformer.k2_matrix.detach().cpu()[0])
        im5 = ax[1][2].imshow(prod_mat2, vmin=prod_mat2.mean() - prod_mat2.std(),
                              vmax=prod_mat2.mean() + prod_mat2.std())
        divider = make_axes_locatable(ax[1][2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im5, cax=cax, orientation='vertical')
        # raise
        ax[0][0].set_title("Query Matrix", fontsize=20)
        ax[0][1].set_title("Key Matrix", fontsize=20)
        ax[0][2].set_title("Matrix Product", fontsize=20)

        ax[0][0].set_ylabel("First Order Operator Matrices", fontsize=20)
        ax[1][0].set_ylabel("Second Order Operator Matrices", fontsize=20)
        # ax[2].imshow(transformer.v_embedding_layer.weight.detach().cpu().T)
        plt.tight_layout()
        plt.savefig("./{}/weight_matrices_{}.png".format(path, seed))
        # plt.show()
    except AttributeError:
        pass

    test_vals = []
    test_value = evaluate(test_loader, transformer, loss_fn)
    test_vals.append(test_value)
    print("TEST VALUE FROM LAST EPOCH: {0:5f}".format(test_value))
    transformer.load_state_dict(torch.load(model_path)['model_state_dict'])
    test_value = evaluate(test_loader, transformer, loss_fn)
    test_vals.append(test_value)
    print("TEST VALUE BEST LAST EPOCH: {0:5f}".format(test_value))
    np.save("{}{}_{}_{}/test_vals_{}.npy".format(config['results_dir'], config['transformer'],
                                                 config['neural_operator'], prefix, seed), test_vals)
    return test_value


if __name__ == '__main__':
    # Create a transformer with an input dimension of 10, a hidden dimension
    # of 20, 2 transformer layers, and 8 attention heads.

    # Load config
    with open("./configs/pitt_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    # Get arguments and get rid of unnecessary ones
    train_args = config['args']
    # file = open("loss_compare.log", "w")
    for batch_size in [
        32,
        # 64,
        # 128
    ]:
        for lr in [
            # 0.001,
            # 1e-4,
            3e-5,
            # 1e-5
        ]:
            for weight_decay in [
                # 0,
                # 0.1,
                # 1e-2,
                # 1e-3,
                # 1e-4,
                # 1e-5,
                # 1e-6,
                1e-8
            ]:
                for dropout in [
                    0,
                    # 0.1,
                    # 0.2
                ]:
                    # train different model with different training set size. Param num_samples=10 means 10 equations(90
                    # frames for each equation) thus 900 data in heat, burgers and KdV respectively. Training rate =
                    # 0.6 thus total training set size is 1620 when num_samples=10.
                    for num_samples in [
                        # 10,
                        100,
                        # 1000
                    ]:
                        prefix = train_args['flnm'] + "_" + train_args['data_name'].split("_")[0] + "_" + train_args[
                            'train_style'] + "_" + \
                                 train_args['embedding']
                        train_args['prefix'] = prefix
                        os.makedirs("{}{}_{}_{}".format(train_args['results_dir'], train_args['transformer'],
                                                        train_args['neural_operator'],
                                                        prefix),
                                    exist_ok=True)
                        shutil.copy("./configs/pitt_config.yaml",
                                    "{}{}_{}_{}/pitt_config.yaml".format(train_args['results_dir'],
                                                                         train_args['transformer'],
                                                                         train_args['neural_operator'], prefix))
                        shutil.copy("./plot_progress.py",
                                    "{}{}_{}_{}/plot_progress.py".format(train_args['results_dir'],
                                                                         train_args['transformer'],
                                                                         train_args['neural_operator'], prefix))
                        # modify super parameters
                        train_args['batch_size'] = batch_size
                        train_args['learning_rate'] = lr
                        train_args['weight_decay'] = weight_decay
                        train_args['dropout'] = dropout
                        train_args['num_samples'] = num_samples
                        val_loss = 0
                        for seed in range(train_args['num_seeds']):
                            # if(seed == 0):
                            #    continue
                            print("\nSEED: {}\n".format(seed))
                            torch.manual_seed(seed)
                            np.random.seed(seed)
                            train_args['seed'] = seed
                            val_loss += run_training(train_args, prefix)
                        val_loss /= train_args['num_seeds']
                        # file.write(
                        #     f"flnm:{train_args['flnm']},data_name:{train_args['data_name']},num_samples:{num_samples},"
                        #     f"batch_size:{batch_size},learning_rate:{lr},weight_decay:{train_args['weight_decay']},"
                        #     f"dropout:{train_args['dropout']}")
                        # file.write(f"loss:{val_loss}")
                        # file.flush()
