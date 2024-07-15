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



