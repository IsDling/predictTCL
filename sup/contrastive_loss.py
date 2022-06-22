import torch

def con_loss(x, pos, neg, tau, ifcondetach):
    if ifcondetach:
        cs_pos = torch.cosine_similarity(x, pos.detach(), dim=1)
        cs_neg = torch.cosine_similarity(x, neg.detach(), dim=1)
    else:
        cs_pos = torch.cosine_similarity(x, pos, dim=1)
        cs_neg = torch.cosine_similarity(x, neg, dim=1)
    loss = torch.mean(-torch.log(torch.exp(cs_pos/tau)/(torch.exp(cs_pos/tau)+torch.exp(cs_neg/tau))))
    return loss

def multi_con_loss(x, pos, neg1, neg2, tau):
    cs_pos = torch.cosine_similarity(x, pos, dim=1)
    cs_neg1 = torch.cosine_similarity(x, neg1, dim=1)
    cs_neg2 = torch.cosine_similarity(x, neg2, dim=1)
    loss = torch.mean(-torch.log(torch.exp(cs_pos/tau)/(torch.exp(cs_pos/tau)+torch.exp(cs_neg1/tau)+torch.exp(cs_neg2/tau))))
    return loss