import re
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from .utils import save, load, plot_prob, savecfvae, savecfvaejointtraining
from .config import JsonConfig
from .models import Glow, CFVAE, CFVAEJointTraining
from . import thops


class Trainer(object):
    def __init__(self, graph, optim, lrschedule, loaded_step,
                 devices, data_device,
                 data, log_dir, hparams):
        if isinstance(hparams, str):
            hparams = JsonConfig(hparams)

        # set members
        # append date info
        self.log_dir = log_dir
        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")

        # write hparams
        hparams.dump(self.log_dir)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        self.checkpoints_gap = hparams.Train.checkpoints_gap
        self.max_checkpoints = hparams.Train.max_checkpoints

        # model relative
        self.graph = graph
        self.optim = optim

        # grad operation
        self.max_grad_clip = hparams.Train.max_grad_clip
        self.max_grad_norm = hparams.Train.max_grad_norm

        # copy devices from built graph
        self.devices = devices
        self.data_device = data_device

        # number of training batches
        self.batch_size = hparams.Train.batch_size
        self.train_dataset = data.get_train_dataset()
        self.data_loader = DataLoader(self.train_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=0,
                                      shuffle=True,
                                      drop_last=True)
                                      
        self.n_epoches = hparams.Train.num_epochs
        self.global_step = 0
        
        self.seqlen = hparams.Data.seqlen
        self.n_lookahead = hparams.Data.n_lookahead
        
        self.val_batch_size = hparams.Train.batch_size

        # test batch
        self.test_data_loader = DataLoader(data.get_test_dataset(),
                                      batch_size=self.batch_size,
                                       num_workers=0,
                                      shuffle=False,
                                      drop_last=True)
        
        self.test_batch = next(iter(self.test_data_loader))
        for k in self.test_batch:
            self.test_batch[k] = self.test_batch[k].to(self.data_device)

        # validation batch
        self.val_data_loader = DataLoader(data.get_validation_dataset(),
                                      batch_size=self.batch_size,
                                      num_workers=0,
                                      shuffle=False,
                                      drop_last=True)
            
        self.data = data
        
        # lr schedule
        self.lrschedule = lrschedule
        self.loaded_step = loaded_step

        # log relative
        # tensorboard
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.scalar_log_gaps = hparams.Train.scalar_log_gap
        self.validation_log_gaps = hparams.Train.validation_log_gap
        self.plot_gaps = hparams.Train.plot_gap
        
    def prepare_cond(self, jt_data, ctrl_data):
        nn,seqlen,n_feats = jt_data.shape
        
        jt_data = jt_data.reshape((nn, seqlen*n_feats))
        nn,seqlen,n_feats = ctrl_data.shape
        ctrl_data = ctrl_data.reshape((nn, seqlen*n_feats))
        cond = torch.from_numpy(np.expand_dims(np.concatenate((jt_data,ctrl_data),axis=1), axis=-1))
        return cond.to(self.data_device)
    
    def generate_sample(self, eps_std=1.0, counter=0):
        print("generate_sample")

        batch = self.test_batch

        autoreg_all = batch["autoreg"].cpu().numpy()
        control_all = batch["control"].cpu().numpy()

        # Initialize the pose sequence with ground truth test data
        seqlen = self.seqlen
        n_lookahead = self.n_lookahead
        
        # Initialize the lstm hidden state
        if hasattr(self.graph, "module"):
            self.graph.module.init_lstm_hidden()
        else:
            self.graph.init_lstm_hidden()
            
        nn,n_timesteps,n_feats = autoreg_all.shape
        sampled_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
        autoreg = np.zeros((nn, seqlen, n_feats), dtype=np.float32) #initialize from a mean pose
        sampled_all[:,:seqlen,:] = autoreg
        

        # Loop through control sequence and generate new data
        for i in range(0,control_all.shape[1]-seqlen-n_lookahead):
            control = control_all[:,i:(i+seqlen+1+n_lookahead),:]
            
            # prepare conditioning for moglow (control + previous poses)
            cond = self.prepare_cond(autoreg.copy(), control.copy())

            # sample from Moglow
            sampled = self.graph(z=None, cond=cond, eps_std=eps_std, reverse=True)
            sampled = sampled.cpu().numpy()[:,:,0]

            # store the sampled frame
            sampled_all[:,(i+seqlen),:] = sampled
            
            # update saved pose sequence
            autoreg = np.concatenate((autoreg[:,1:,:].copy(), sampled[:,None,:]), axis=1)
            
        
        # store the generated animations
        self.data.save_animation(control_all[:,:(n_timesteps-n_lookahead),:], sampled_all, os.path.join(self.log_dir, f'sampled_{counter}_temp{str(int(eps_std*100))}_{str(self.global_step//1000)}k'))              
    
    def count_parameters(self, model):
         return sum(p.numel() for p in model.parameters() if p.requires_grad)    

    def train(self):

        self.global_step = self.loaded_step

        # begin to train
        for epoch in range(self.n_epoches):
            print("epoch", epoch, "global step", self.global_step)
            progress = tqdm(self.data_loader)
            for i_batch, batch in enumerate(progress):

                # set to training state
                self.graph.train()
                
                # update learning rate
                lr = self.lrschedule["func"](global_step=self.global_step,
                                             **self.lrschedule["args"])
                                                             
                for param_group in self.optim.param_groups:
                    param_group['lr'] = lr
                self.optim.zero_grad()
                if self.global_step % self.scalar_log_gaps == 0:
                    self.writer.add_scalar("lr/lr", lr, self.global_step)
                    
                # get batch data
                for k in batch:
                    batch[k] = batch[k].to(self.data_device)
                x = batch["x"]
                                
                cond = batch["cond"]

                # init LSTM hidden
                if hasattr(self.graph, "module"):
                    self.graph.module.init_lstm_hidden()
                else:
                    self.graph.init_lstm_hidden()

                # at first time, initialize ActNorm
                if self.global_step == 0:
                    self.graph(x[:self.batch_size // len(self.devices), ...],
                               cond[:self.batch_size // len(self.devices), ...] if cond is not None else None)
                    # re-init LSTM hidden
                    if hasattr(self.graph, "module"):
                        self.graph.module.init_lstm_hidden()
                    else:
                        self.graph.init_lstm_hidden()
                
                #print("n_params: " + str(self.count_parameters(self.graph)))
                
                # parallel
                if len(self.devices) > 1 and not hasattr(self.graph, "module"):
                    print("[Parallel] move to {}".format(self.devices))
                    self.graph = torch.nn.parallel.DataParallel(self.graph, self.devices, self.devices[0])
                    
                
                # forward phase
                z, nll = self.graph(x=x, cond=cond)

                # loss
                loss_generative = Glow.loss_generative(nll)
                loss_classes = 0
                if self.global_step % self.scalar_log_gaps == 0:
                    self.writer.add_scalar("loss/loss_generative", loss_generative, self.global_step)
                loss = loss_generative

                # backward
                self.graph.zero_grad()
                self.optim.zero_grad()
                loss.backward()

                # operate grad
                if self.max_grad_clip is not None and self.max_grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(self.graph.parameters(), self.max_grad_clip)
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.max_grad_norm)
                    if self.global_step % self.scalar_log_gaps == 0:
                        self.writer.add_scalar("grad_norm/grad_norm", grad_norm, self.global_step)
                # step
                self.optim.step()

                if self.global_step % self.validation_log_gaps == 0:
                    # set to eval state
                    self.graph.eval()

                                        
                    # Validation forward phase
                    loss_val = 0
                    n_batches = 0
                    for ii, val_batch in enumerate(self.val_data_loader):
                        for k in val_batch:
                            val_batch[k] = val_batch[k].to(self.data_device)
                            
                        with torch.no_grad():
                            
                            # init LSTM hidden
                            if hasattr(self.graph, "module"):
                                self.graph.module.init_lstm_hidden()
                            else:
                                self.graph.init_lstm_hidden()
                                
                            z_val, nll_val = self.graph(x=val_batch["x"], cond=val_batch["cond"])
                            
                            # loss
                            loss_val = loss_val + Glow.loss_generative(nll_val)
                            n_batches = n_batches + 1        
                    
                    loss_val = loss_val/n_batches
                    self.writer.add_scalar("val_loss/val_loss_generative", loss_val, self.global_step)
                    
                                
                # checkpoints
                if self.global_step % self.checkpoints_gap == 0 and self.global_step > 0:
                    save(global_step=self.global_step,
                         graph=self.graph,
                         optim=self.optim,
                         pkg_dir=self.checkpoints_dir,
                         is_best=True,
                         max_checkpoints=self.max_checkpoints)
                         
                # generate samples and save
                if self.global_step % self.plot_gaps == 0 and self.global_step > 0:   
                    self.generate_sample(eps_std=1.0)

                # global step
                self.global_step += 1
            print(
                f'Loss: {loss.item():.5f}/ Validation Loss: {loss_val:.5f} '
            )

        self.writer.export_scalars_to_json(os.path.join(self.log_dir, "all_scalars.json"))
        self.writer.close()


class TrainerCFVAE(object):
    def __init__(self, graph, aeoptim, flowoptim, aelrschedule, flowlrschedule, loaded_step,
                 devices, data_device,
                 data, log_dir, hparams):
        if isinstance(hparams, str):
            hparams = JsonConfig(hparams)

        # set members
        # append date info
        self.log_dir = log_dir
        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")

        # write hparams
        hparams.dump(self.log_dir)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        self.checkpoints_gap = hparams.Train.checkpoints_gap
        self.max_checkpoints = hparams.Train.max_checkpoints

        # model relative
        self.graph = graph
        self.aeoptim = aeoptim
        self.flowoptim = flowoptim

        # grad operation
        self.max_grad_clip = hparams.Train.max_grad_clip
        self.max_grad_norm = hparams.Train.max_grad_norm

        # copy devices from built graph
        self.devices = devices
        self.data_device = data_device

        # number of training batches
        self.batch_size = hparams.Train.batch_size
        self.train_dataset = data.get_train_dataset()
        self.data_loader = DataLoader(self.train_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=0,
                                      shuffle=True,
                                      drop_last=True)
                                      
        self.n_epoches = hparams.Train.num_epochs
        self.n_ae_loops = hparams.Train.num_ae_loops
        self.n_flow_loops = hparams.Train.num_flow_loops

        self.global_step = 0
        
        self.seqlen = hparams.Data.seqlen
        self.n_lookahead = hparams.Data.n_lookahead
        
        self.val_batch_size = hparams.Train.batch_size

        # test batch
        self.test_data_loader = DataLoader(data.get_test_dataset(),
                                      batch_size=self.batch_size,
                                       num_workers=0,
                                      shuffle=False,
                                      drop_last=True)
        
        self.test_batch = next(iter(self.test_data_loader))
        for k in self.test_batch:
            self.test_batch[k] = self.test_batch[k].to(self.data_device)

        # validation batch
        self.val_data_loader = DataLoader(data.get_validation_dataset(),
                                      batch_size=self.batch_size,
                                      num_workers=0,
                                      shuffle=False,
                                      drop_last=True)
            
        self.data = data
        
        # lr schedule
        self.aelrschedule = aelrschedule
        self.flowlrschedule = flowlrschedule
        self.loaded_step = loaded_step

        # log relative
        # tensorboard
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.scalar_log_gaps = hparams.Train.scalar_log_gap
        self.validation_log_gaps = hparams.Train.validation_log_gap
        self.plot_gaps = hparams.Train.plot_gap
        
    def prepare_cond(self, jt_data, ctrl_data):
        nn,seqlen,n_feats = jt_data.shape
        
        jt_data = jt_data.reshape((nn, seqlen*n_feats))
        nn,seqlen,n_feats = ctrl_data.shape
        ctrl_data = ctrl_data.reshape((nn, seqlen*n_feats))
        cond = torch.from_numpy(np.expand_dims(np.concatenate((jt_data,ctrl_data),axis=1), axis=-1))
        return cond.to(self.data_device)
    

    def test_autoencoder(self):
        train_batch = next(iter(self.data_loader))

        for k in train_batch:
            train_batch[k] = train_batch[k].to(self.data_device)
        x = train_batch["x"]

        with torch.no_grad():
            y,z = self.graph(x=x)
            y = y.permute(0,2,1).cpu().numpy()
            z = z.permute(0,2,1).cpu().numpy()
        
        # make histogram of first 
        
        self.data.save_animation(None, y, '/home/w0457094/git/StyleGestures/aetest/test')              


    def generate_sample(self, eps_std=1.0, counter=0):
        print("generate_sample")

        batch = self.test_batch

        autoreg_all = batch["autoreg"].cpu().numpy()
        control_all = batch["control"].cpu().numpy()

        # Initialize the pose sequence with ground truth test data
        seqlen = self.seqlen
        n_lookahead = self.n_lookahead
        
        # Initialize the lstm hidden state
        if hasattr(self.graph, "module"):
            self.graph.module.init_lstm_hidden()
        else:
            self.graph.init_lstm_hidden()
            
        nn,n_timesteps,n_feats = autoreg_all.shape
        sampled_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
        autoreg = np.zeros((nn, seqlen, n_feats), dtype=np.float32) #initialize from a mean pose
        sampled_all[:,:seqlen,:] = autoreg
        

        # Loop through control sequence and generate new data
        for i in range(0,control_all.shape[1]-seqlen-n_lookahead):
            control = control_all[:,i:(i+seqlen+1+n_lookahead),:]
            
            # prepare conditioning for moglow (control + previous poses)
            cond = self.prepare_cond(autoreg.copy(), control.copy())

            # sample from CFVAE
            sampled = self.graph(z=None, cond=cond, eps_std=eps_std, reverse=True)

            sampled = sampled.cpu().numpy()[:,:,0]
            # store the sampled frame
            sampled_all[:,(i+seqlen),:] = sampled

            # update saved pose sequence
            autoreg = np.concatenate((autoreg[:,1:,:].copy(), sampled[:,None,:]), axis=1)
            
        
        # store the generated animations
        self.data.save_animation(control_all[:,:(n_timesteps-n_lookahead),:], sampled_all, os.path.join(self.log_dir, f'sampled_{counter}_temp{str(int(eps_std*100))}_{str(self.global_step)}k'))              
    
    def count_parameters(self, model):
         return sum(p.numel() for p in model.parameters() if p.requires_grad)    

    def train(self):

        self.global_step = self.loaded_step

        init_actnorm = True
        # train
        # adjusted_n_epochs = self.n_epoches // (self.n_ae_loops + self.n_flow_loops)
        for epoch in range(self.n_epoches):
            print("epoch", epoch, "global step", self.global_step)
            # at first time, re-init LSTM hidden
            if self.global_step == 0:
                # re-init LSTM hidden
                if hasattr(self.graph, "module"):
                    self.graph.module.init_lstm_hidden()
                else:
                    self.graph.init_lstm_hidden()

            for ae_epoch in range(self.n_ae_loops):
                progress = tqdm(self.data_loader)
                for i_batch, batch in enumerate(progress):

                    # set autoencoder to training state
                    self.graph.encoder.train()
                    self.graph.decoder.train()
                    
                    # update learning rate
                    lr = self.aelrschedule["func"](global_step=(self.global_step+1)*(ae_epoch+1),
                                                **self.aelrschedule["args"])
                                                                
                    for param_group in self.aeoptim.param_groups:
                        param_group['lr'] = lr
                    self.aeoptim.zero_grad()
                    if self.global_step % self.scalar_log_gaps == 0:
                        self.writer.add_scalar("lr/lr", lr, self.global_step)
                        
                    # get batch data
                    for k in batch:
                        batch[k] = batch[k].to(self.data_device)
                    x = batch["x"]
                                    
                    # init LSTM hidden
                    if hasattr(self.graph, "module"):
                        self.graph.module.init_lstm_hidden()
                    else:
                        self.graph.init_lstm_hidden()

                    # forward phase
                    y, _ = self.graph(x=x)

                    # loss
                    ae_loss = CFVAE.loss_generative(y=y,x=x)
                    if self.global_step % self.scalar_log_gaps == 0:
                        self.writer.add_scalar("loss/ae_loss_generative", ae_loss, self.global_step)

                    # backward
                    self.graph.zero_grad()
                    self.aeoptim.zero_grad()
                    ae_loss.backward()

                    # operate grad
                    if self.max_grad_clip is not None and self.max_grad_clip > 0:
                        torch.nn.utils.clip_grad_value_(self.graph.parameters(), self.max_grad_clip)
                    if self.max_grad_norm is not None and self.max_grad_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.max_grad_norm)
                        if self.global_step % self.scalar_log_gaps == 0:
                            self.writer.add_scalar("grad_norm/grad_norm", grad_norm, self.global_step)
                    # step
                    self.aeoptim.step()
                
                print(
                f'{ae_epoch} AE Loss: {ae_loss.item():.5f} (lr: {lr})'
                )

            # print('testing autoencoder')
            # self.test_autoencoder()
            # checkpoints
            if self.global_step % self.checkpoints_gap == 0:
                savecfvae(global_step=self.global_step,
                        graph=self.graph,
                        aeoptim=self.aeoptim,
                        flowoptim=self.flowoptim,
                        pkg_dir=self.checkpoints_dir,
                        is_best=True,
                        max_checkpoints=self.max_checkpoints)

            for flow_epoch in range(self.n_flow_loops):
                progress = tqdm(self.data_loader)
                for i_batch, batch in enumerate(progress):

                    # set to training state
                    self.graph.flow.train()
                    
                    # update learning rate
                    lr = self.flowlrschedule["func"](global_step=(self.global_step+1)*(flow_epoch+1),
                                                **self.flowlrschedule["args"])
                                                                
                    for param_group in self.flowoptim.param_groups:
                        param_group['lr'] = lr
                    self.flowoptim.zero_grad()
                    if self.global_step % self.scalar_log_gaps == 0:
                        self.writer.add_scalar("lr/lr", lr, self.global_step)
                        
                    # get batch data
                    for k in batch:
                        batch[k] = batch[k].to(self.data_device)
                    x = batch["x"]
                    cond = batch["cond"]

                    # init LSTM hidden
                    if hasattr(self.graph, "module"):
                        self.graph.module.init_lstm_hidden()
                    else:
                        self.graph.init_lstm_hidden()

                    # at first time, initialize ActNorm
                    if init_actnorm:
                        init_actnorm = False
                        self.graph(x[:self.batch_size // len(self.devices), ...],
                                cond[:self.batch_size // len(self.devices), ...] if cond is not None else None, train_flow=True)
                        # re-init LSTM hidden
                        if hasattr(self.graph, "module"):
                            self.graph.module.init_lstm_hidden()
                        else:
                            self.graph.init_lstm_hidden()
                                        
                    
                    # forward phase
                    nll = self.graph(x=x, cond=cond, train_flow=True)

                    # loss
                    flow_loss = CFVAE.loss_generative(nll=nll)
                    if self.global_step % self.scalar_log_gaps == 0:
                        self.writer.add_scalar("loss/flow_loss_generative", flow_loss, self.global_step)

                    # backward
                    self.graph.zero_grad()
                    self.flowoptim.zero_grad()
                    flow_loss.backward()

                    # operate grad
                    if self.max_grad_clip is not None and self.max_grad_clip > 0:
                        torch.nn.utils.clip_grad_value_(self.graph.parameters(), self.max_grad_clip)
                    if self.max_grad_norm is not None and self.max_grad_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.max_grad_norm)
                        if self.global_step % self.scalar_log_gaps == 0:
                            self.writer.add_scalar("grad_norm/grad_norm", grad_norm, self.global_step)
                    # step
                    self.flowoptim.step()
                print(
                f'{flow_epoch} Flow Loss: {flow_loss.item():.5f} (lr: {lr})'
                )

                # checkpoints
                if flow_epoch % self.checkpoints_gap == 0:
                    savecfvae(global_step=self.global_step,
                            graph=self.graph,
                            aeoptim=self.aeoptim,
                            flowoptim=self.flowoptim,
                            pkg_dir=self.checkpoints_dir,
                            is_best=True,
                            max_checkpoints=self.max_checkpoints)
                            
                # generate samples and save
                if flow_epoch % self.plot_gaps == 0: 
                    self.generate_sample(eps_std=1.0)

            # validation
            if self.global_step % self.validation_log_gaps == 0:
                # set to eval state
                self.graph.eval()
                                    
                # Validation forward phase
                mse_loss_val = 0
                nll_loss_val = 0
                n_batches = 0
                for ii, val_batch in enumerate(self.val_data_loader):
                    for k in val_batch:
                        val_batch[k] = val_batch[k].to(self.data_device)
                        
                    with torch.no_grad():
                        
                        # init LSTM hidden
                        if hasattr(self.graph, "module"):
                            self.graph.module.init_lstm_hidden()
                        else:
                            self.graph.init_lstm_hidden()
                            
                        y_val,_ = self.graph(x=val_batch["x"], cond=val_batch["cond"])
                        nll_val = self.graph(x=val_batch["x"], cond=val_batch["cond"], train_flow=True)
                        
                        # loss
                        # loss_val = loss_val + CFVAE.loss_generative(nll_val)
                        mse_loss_val = mse_loss_val + CFVAE.loss_generative(y=y_val, x=val_batch["x"])
                        nll_loss_val = nll_loss_val + CFVAE.loss_generative(nll=nll_val)
                        n_batches = n_batches + 1        
                
                mse_loss_val = mse_loss_val/n_batches
                nll_loss_val = nll_loss_val/n_batches
                self.writer.add_scalar("val_loss/val_mse_loss_generative", mse_loss_val, self.global_step)
                self.writer.add_scalar("val_loss/val_nll_loss_generative", nll_loss_val, self.global_step)
                
            print(
                f'Val Losses: {mse_loss_val.item():.5f}, {nll_loss_val.item():.5f}'
            )
                            
            # checkpoints
            if self.global_step % self.checkpoints_gap == 0:
                savecfvae(global_step=self.global_step,
                        graph=self.graph,
                        aeoptim=self.aeoptim,
                        flowoptim=self.flowoptim,
                        pkg_dir=self.checkpoints_dir,
                        is_best=True,
                        max_checkpoints=self.max_checkpoints)
                        
            # generate samples and save
            if self.global_step % self.plot_gaps == 0: 
                self.generate_sample(eps_std=1.0)

            # global step
            self.global_step += 1


        # # final flow optimisation
        # for flow_epoch in range(100):
        #     progress = tqdm(self.data_loader)
        #     for i_batch, batch in enumerate(progress):

        #         # set to training state
        #         self.graph.flow.train()
                
        #         # update learning rate
        #         lr = self.flowlrschedule["func"](global_step=self.global_step,
        #                                     **self.flowlrschedule["args"])
                                                            
        #         for param_group in self.flowoptim.param_groups:
        #             param_group['lr'] = lr
        #         self.flowoptim.zero_grad()
        #         if self.global_step % self.scalar_log_gaps == 0:
        #             self.writer.add_scalar("lr/lr", lr, self.global_step)
                    
        #         # get batch data
        #         for k in batch:
        #             batch[k] = batch[k].to(self.data_device)
        #         x = batch["x"]
        #         cond = batch["cond"]

        #         # init LSTM hidden
        #         if hasattr(self.graph, "module"):
        #             self.graph.module.init_lstm_hidden()
        #         else:
        #             self.graph.init_lstm_hidden()
                
        #         # forward phase
        #         nll = self.graph(x=x, cond=cond, train_flow=True)

        #         # loss
        #         flow_loss_generative = CFVAE.loss_generative(nll=nll)
        #         loss_classes = 0
        #         if self.global_step % self.scalar_log_gaps == 0:
        #             self.writer.add_scalar("loss/flow_loss_generative", flow_loss_generative, self.global_step)
        #         flow_loss = flow_loss_generative

        #         # backward
        #         self.graph.zero_grad()
        #         self.flowoptim.zero_grad()
        #         flow_loss.backward()

        #         # operate grad
        #         if self.max_grad_clip is not None and self.max_grad_clip > 0:
        #             torch.nn.utils.clip_grad_value_(self.graph.parameters(), self.max_grad_clip)
        #         if self.max_grad_norm is not None and self.max_grad_norm > 0:
        #             grad_norm = torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.max_grad_norm)
        #             if self.global_step % self.scalar_log_gaps == 0:
        #                 self.writer.add_scalar("grad_norm/grad_norm", grad_norm, self.global_step)
        #         # step
        #         self.flowoptim.step()
        #     print(
        #     f'Flow Loss (final optimisation): {flow_loss.item():.5f}'
        #     )

        # save final checkpoint
        savecfvae(global_step=self.global_step,
                graph=self.graph,
                aeoptim=self.aeoptim,
                flowoptim=self.flowoptim,
                pkg_dir=self.checkpoints_dir,
                is_best=True,
                max_checkpoints=self.max_checkpoints)
                
        # generate samples and save
        self.generate_sample(eps_std=1.0)

        self.writer.export_scalars_to_json(os.path.join(self.log_dir, "all_scalars.json"))
        self.writer.close()

class TrainerCFVAEJointTraining(object):
    def __init__(self, graph, optim, lrschedule, loaded_step,
                 devices, data_device,
                 data, log_dir, hparams):
        if isinstance(hparams, str):
            hparams = JsonConfig(hparams)

        # set members
        # append date info
        self.log_dir = log_dir
        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")

        # write hparams
        hparams.dump(self.log_dir)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        self.checkpoints_gap = hparams.Train.checkpoints_gap
        self.max_checkpoints = hparams.Train.max_checkpoints

        # model relative
        self.graph = graph
        self.optim = optim

        # grad operation
        self.max_grad_clip = hparams.Train.max_grad_clip
        self.max_grad_norm = hparams.Train.max_grad_norm

        # copy devices from built graph
        self.devices = devices
        self.data_device = data_device

        # number of training batches
        self.batch_size = hparams.Train.batch_size
        self.train_dataset = data.get_train_dataset()
        self.data_loader = DataLoader(self.train_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=0,
                                      shuffle=True,
                                      drop_last=True)
                                      
        self.n_epoches = hparams.Train.num_epochs

        self.global_step = 0
        
        self.seqlen = hparams.Data.seqlen
        self.n_lookahead = hparams.Data.n_lookahead
        
        self.val_batch_size = hparams.Train.batch_size

        # test batch
        self.test_data_loader = DataLoader(data.get_test_dataset(),
                                      batch_size=self.batch_size,
                                       num_workers=0,
                                      shuffle=False,
                                      drop_last=True)
        
        self.test_batch = next(iter(self.test_data_loader))
        for k in self.test_batch:
            self.test_batch[k] = self.test_batch[k].to(self.data_device)

        # validation batch
        self.val_data_loader = DataLoader(data.get_validation_dataset(),
                                      batch_size=self.batch_size,
                                      num_workers=0,
                                      shuffle=False,
                                      drop_last=True)
            
        self.data = data
        
        # lr schedule
        self.lrschedule = lrschedule
        self.loaded_step = loaded_step

        # log relative
        # tensorboard
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.scalar_log_gaps = hparams.Train.scalar_log_gap
        self.validation_log_gaps = hparams.Train.validation_log_gap
        self.plot_gaps = hparams.Train.plot_gap
        
    def prepare_cond(self, jt_data, ctrl_data):
        nn,seqlen,n_feats = jt_data.shape
        
        jt_data = jt_data.reshape((nn, seqlen*n_feats))
        nn,seqlen,n_feats = ctrl_data.shape
        ctrl_data = ctrl_data.reshape((nn, seqlen*n_feats))
        cond = torch.from_numpy(np.expand_dims(np.concatenate((jt_data,ctrl_data),axis=1), axis=-1))
        return cond.to(self.data_device)
    

    def test_autoencoder(self):
        train_batch = next(iter(self.data_loader))

        for k in train_batch:
            train_batch[k] = train_batch[k].to(self.data_device)
        x = train_batch["x"]

        with torch.no_grad():
            y,z = self.graph(x=x)
            y = y.permute(0,2,1).cpu().numpy()
            z = z.permute(0,2,1).cpu().numpy()
        
        # make histogram of first 
        
        self.data.save_animation(None, y, '/home/w0457094/git/StyleGestures/aetest/test')              


    def generate_sample(self, eps_std=1.0, counter=0):
        print("generate_sample")

        batch = self.test_batch

        autoreg_all = batch["autoreg"].cpu().numpy()
        control_all = batch["control"].cpu().numpy()

        # Initialize the pose sequence with ground truth test data
        seqlen = self.seqlen
        n_lookahead = self.n_lookahead
        
        # Initialize the lstm hidden state
        if hasattr(self.graph, "module"):
            self.graph.module.init_lstm_hidden()
        else:
            self.graph.init_lstm_hidden()
            
        nn,n_timesteps,n_feats = autoreg_all.shape
        sampled_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
        autoreg = np.zeros((nn, seqlen, n_feats), dtype=np.float32) #initialize from a mean pose
        sampled_all[:,:seqlen,:] = autoreg
        

        # Loop through control sequence and generate new data
        for i in range(0,control_all.shape[1]-seqlen-n_lookahead):
            control = control_all[:,i:(i+seqlen+1+n_lookahead),:]
            
            # prepare conditioning for moglow (control + previous poses)
            cond = self.prepare_cond(autoreg.copy(), control.copy())

            # sample from CFVAE
            sampled = self.graph(z=None, cond=cond, eps_std=eps_std, reverse=True)

            sampled = sampled.cpu().numpy()[:,:,0]
            # store the sampled frame
            sampled_all[:,(i+seqlen),:] = sampled

            # update saved pose sequence
            autoreg = np.concatenate((autoreg[:,1:,:].copy(), sampled[:,None,:]), axis=1)
            
        
        # store the generated animations
        self.data.save_animation(control_all[:,:(n_timesteps-n_lookahead),:], sampled_all, os.path.join(self.log_dir, f'sampled_{counter}_temp{str(int(eps_std*100))}_{str(self.global_step)}'))              
    
    def count_parameters(self, model):
         return sum(p.numel() for p in model.parameters() if p.requires_grad)    

    def train(self):

        self.global_step = self.loaded_step

        init_actnorm = True
        # train
        for epoch in range(self.n_epoches):
            print("epoch", epoch, "global step", self.global_step)
            # at first time, re-init LSTM hidden
            if self.global_step == 0:
                # re-init LSTM hidden
                if hasattr(self.graph, "module"):
                    self.graph.module.init_lstm_hidden()
                else:
                    self.graph.init_lstm_hidden()

            progress = tqdm(self.data_loader)
            for i_batch, batch in enumerate(progress):

                # set to training state
                self.graph.train()
                
                # update learning rate
                lr = self.lrschedule["func"](global_step=self.global_step,
                                            **self.lrschedule["args"])
                                                            
                for param_group in self.optim.param_groups:
                    param_group['lr'] = lr
                self.optim.zero_grad()
                if self.global_step % self.scalar_log_gaps == 0:
                    self.writer.add_scalar("lr/lr", lr, self.global_step)
                    
                # get batch data
                for k in batch:
                    batch[k] = batch[k].to(self.data_device)
                x = batch["x"]
                cond = batch["cond"]
       
                # init LSTM hidden
                if hasattr(self.graph, "module"):
                    self.graph.module.init_lstm_hidden()
                else:
                    self.graph.init_lstm_hidden()

                # forward phase
                nll,y = self.graph(x=x, cond=cond)

                # loss
                # loss, nll_mean, recloss = CFVAEJointTraining.loss_generative(nll=nll,y=self.data.evaluate_joints(y),x=self.data.evaluate_joints(x))
                loss, nll_mean, recloss = self.graph.loss_generative(nll=nll,y=y,x=x)

                if self.global_step % self.scalar_log_gaps == 0:
                    self.writer.add_scalar("loss/loss_generative", loss, self.global_step)

                # backward
                self.graph.zero_grad()
                self.optim.zero_grad()
                loss.backward()

                # operate grad
                if self.max_grad_clip is not None and self.max_grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(self.graph.parameters(), self.max_grad_clip)
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.max_grad_norm)
                    if self.global_step % self.scalar_log_gaps == 0:
                        self.writer.add_scalar("grad_norm/grad_norm", grad_norm, self.global_step)
                # step
                self.optim.step()
                
                # global step
                self.global_step += 1
                
                # checkpoints
                if self.global_step % self.checkpoints_gap == 0 and self.global_step>0:
                    savecfvaejointtraining(global_step=self.global_step,
                            graph=self.graph,
                            optim=self.optim,
                            pkg_dir=self.checkpoints_dir,
                            is_best=True,
                            max_checkpoints=self.max_checkpoints)

                # generate samples and save
                if self.global_step % self.plot_gaps == 0 and self.global_step>0: 
                    self.generate_sample(eps_std=1.0)

            # validation
            if self.global_step % self.validation_log_gaps == 0:
                # set to eval state
                self.graph.eval()
                                    
                # Validation forward phase
                mse_loss_val = 0
                nll_loss_val = 0
                loss_val = 0

                n_batches = 0
                for ii, val_batch in enumerate(self.val_data_loader):
                    for k in val_batch:
                        val_batch[k] = val_batch[k].to(self.data_device)
                        
                    with torch.no_grad():
                        
                        # init LSTM hidden
                        if hasattr(self.graph, "module"):
                            self.graph.module.init_lstm_hidden()
                        else:
                            self.graph.init_lstm_hidden()
                            
                        nll_val, y_val = self.graph(x=val_batch["x"], cond=val_batch["cond"])
                        
                        # loss
                        # loss_v, loss_v_nll, loss_v_rec = CFVAEJointTraining.loss_generative(nll=nll_val,y=self.data.evaluate_joints(y_val), x=self.data.evaluate_joints(val_batch["x"]))
                        loss_v, loss_v_nll, loss_v_rec = self.graph.loss_generative(nll=nll_val,y=y_val, x=val_batch["x"])

                        mse_loss_val = mse_loss_val + loss_v_rec
                        nll_loss_val = nll_loss_val + loss_v_nll
                        loss_val = loss_val + loss_v
                        n_batches = n_batches + 1        
                
                mse_loss_val = mse_loss_val/n_batches
                nll_loss_val = nll_loss_val/n_batches
                loss_val = loss_val/n_batches
                self.writer.add_scalar("val_loss/val_mse_loss_generative", mse_loss_val, self.global_step)
                self.writer.add_scalar("val_loss/val_nll_loss_generative", nll_loss_val, self.global_step)
                self.writer.add_scalar("val_loss/val_loss_generative", loss_val, self.global_step)
                
                print(
                    f'Val Losses: {loss_val.item():.5f}, {mse_loss_val.item():.5f}, {nll_loss_val.item():.5f}'
                )
                    
            print(
            f'{epoch} Loss: {loss.item():.5f} (lr: {lr:.5f}, nll: {nll_mean.item():.5f}, recloss: {recloss.item():.5f})'
            )


        # save final model
        savecfvaejointtraining(global_step=self.global_step,
                graph=self.graph,
                optim=self.optim,
                pkg_dir=self.checkpoints_dir,
                is_best=True,
                max_checkpoints=self.max_checkpoints)
                
        # generate final samples and save
        self.generate_sample(eps_std=1.0)

        self.writer.export_scalars_to_json(os.path.join(self.log_dir, "all_scalars.json"))
        self.writer.close()
