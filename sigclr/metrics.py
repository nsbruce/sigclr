import torch

def get_kl_divergence_ws_distance(le, orig_df_test, device, best_model, orig_X):
    num_of_cell_types=len(le.classes_)
    uniform_prob=1/num_of_cell_types*torch.ones(num_of_cell_types,device=device)
    cum_uniform=torch.cumsum(uniform_prob,0)
    res_unknwn_df=orig_df_test[['yy','yy_int']].copy()
    #for X_test in orig_X.split(batch_size): we might need to batch this if query cannot fit on GPU
    our_out_probs=best_model.pred_prob(orig_X.to(device))
    our_out_cumprobs=our_out_probs.cumsum(axis=1) #reshape(-1).cpu().numpy()

    # KL divergence metrics and Wasserstein distnace.
    kl_divergence=(our_out_probs*torch.log(our_out_probs/uniform_prob+1e-10)).sum(axis=1).cpu().numpy()
    ws_distance= (our_out_cumprobs-cum_uniform).abs().sum(axis=1).cpu().numpy()
    return kl_divergence, ws_distance



def ntXent_loss(batch):
    (xi,xj), _ = batch
    #if xi.shape[0]!=self.batch_size: # Recompute the mask
    self.batch_size=xi.shape[0]
    self.N=self.batch_size*self.world_size 
    self.allN=2*self.N #The batch is effectively 2*batch_size*number_of_GPUs batchs from the GPUs/processes
    
    self.mask = torch.ones((self.allN, self.allN), dtype=bool,device=self.device)
    self.mask = self.mask.fill_diagonal_(0)
    for i in range(self.N):
        self.mask[i, self.N + i] = 0
        self.mask[self.N + i, i] = 0
        
    zi,zj, hi, hj = self.forward(xi,xj)

# Gather and sync the rest of minibatches results:
    if self.world_size > 1:
        z_i = torch.cat(BatchSync.apply(z_i), dim=0)
        z_j = torch.cat(BatchSync.apply(z_j), dim=0)
    z = torch.cat((zi, zj), dim=0)
    
    sim = self.similarity(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

    sim_i_j = torch.diag(sim, self.N)
    sim_j_i = torch.diag(sim, -self.N)

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(self.allN, 1)
    negative_samples = sim[self.mask].reshape(self.allN, -1)

    labels = torch.zeros(self.allN).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    loss = self.criterion(logits, labels)
    loss /= self.allN

    return loss
