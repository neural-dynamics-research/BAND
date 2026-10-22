import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F

class ReverseGradient(Function):
    @staticmethod
    def forward(ctx,x,scale): ctx.scale=scale;return x.view_as(x)
    @staticmethod
    def backward(ctx,g): return -ctx.scale*g,None

class BAND(nn.Module):
    def __init__(self,channels,hidden=32,latent=16,n_subjects=16,dropout=.1,transition=True,subject_effect=True,adversary=True,probabilistic=True):
        super().__init__();self.latent=latent;self.transition=transition
        self.subject_effect=subject_effect;self.adversary=adversary;self.probabilistic=probabilistic
        self.encoder=nn.Sequential(nn.Conv1d(channels,hidden,7,padding=3),nn.GELU(),nn.AvgPool1d(4),nn.Conv1d(hidden,hidden,5,padding=2),nn.GELU())
        self.spectral=nn.Sequential(nn.Linear(channels * 3,hidden),nn.GELU())
        self.fuse=nn.Linear(2*hidden,hidden)
        self.gru=nn.GRU(hidden,hidden,batch_first=True)
        self.z_mu=nn.Linear(hidden,latent);self.z_logvar=nn.Linear(hidden,latent)
        self.u_mu=nn.Linear(hidden,latent);self.u_logvar=nn.Linear(hidden,latent)
        self.head=nn.Sequential(nn.Linear(2*latent,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,2))
        self.id_head=nn.Linear(latent,n_subjects)
        # Auxiliary spectral consistency: shared dynamics predict log beta power.
        self.beta_head=nn.Sequential(nn.Linear(latent,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.prior_mu=nn.Linear(latent+hidden,latent)
        self.prior_logvar=nn.Linear(latent+hidden,latent)
    @staticmethod
    def draw(mu,lv,sample):
        return mu+torch.randn_like(mu)*torch.exp(.5*lv) if sample else mu
    def forward(self,x,grl=1.,sample=True,new_subject=False):
        temporal=self.encoder(x).transpose(1,2)
        fft=torch.fft.rfft(x,dim=-1).abs()
        freq=torch.fft.rfftfreq(x.shape[-1],d=1/250.).to(x.device)
        bands=[]
        for low,high in ((4.,8.),(8.,13.),(13.,30.)):
            mask=(freq>=low)&(freq<high)
            bands.append(fft[:,:,mask].mean(-1) if mask.any() else fft.new_zeros(fft.shape[:2]))
        spectral=self.spectral(torch.cat(bands,dim=1))
        h=self.fuse(torch.cat([temporal,spectral[:,None,:].expand(-1,temporal.shape[1],-1)],dim=-1))
        if self.transition:
            h,_=self.gru(h)
        else:
            h=h.mean(1,keepdim=True)
        # Time-indexed diagonal-Gaussian posteriors.
        step_mu=self.z_mu(h)
        step_lv=torch.clamp(self.z_logvar(h),-8,5)
        if self.transition:
            prev=torch.cat([torch.zeros_like(step_mu[:,:1]),step_mu[:,:-1]],dim=1)
            cond=torch.cat([prev,h],dim=-1)
            prior_mu=self.prior_mu(cond)
            prior_lv=torch.clamp(self.prior_logvar(cond),-8,5)
        else:
            prior_mu=torch.zeros_like(step_mu)
            prior_lv=torch.zeros_like(step_lv)
        zm=step_mu.mean(1)
        zl=torch.logsumexp(step_lv,dim=1)-torch.log(step_lv.new_tensor(step_lv.shape[1]))
        um=self.u_mu(h.mean(1));ul=torch.clamp(self.u_logvar(h.mean(1)),-8,5)
        z=self.draw(zm,zl,sample and self.probabilistic)
        if not self.subject_effect: u=torch.zeros_like(z)
        elif new_subject: u=torch.randn_like(z) if sample and self.probabilistic else torch.zeros_like(z)
        else: u=self.draw(um,ul,sample and self.probabilistic)
        logits=self.head(torch.cat([z,u],-1))
        # Band-power targets are computed exclusively from the current input window.
        # x has already undergone train-fold standardization; no label information is used.
        beta_mask=(freq>=13.)&(freq<30.)
        beta_target=torch.log1p(fft[:,:,beta_mask].square().mean(dim=(1,2))).detach() if beta_mask.any() else x.new_zeros((x.shape[0],))
        beta_pred=self.beta_head(zm).squeeze(-1)
        ident=self.id_head(ReverseGradient.apply(z,grl))
        return {'logits':logits,'zm':zm,'zl':zl,'um':um,'ul':ul,'z':z,'u':u,
                'id_logits':ident,'step_mu':step_mu,'step_lv':step_lv,
                'prior_mu':prior_mu,'prior_lv':prior_lv,
                'beta_target':beta_target,'beta_pred':beta_pred}
    def predict_new(self,x,draws=16):
        """MC integration of q(z|X) p(u), not subject-label inference."""
        self.eval()
        with torch.no_grad():
            pred=[self.forward(x,sample=True,new_subject=True)['logits'].softmax(-1) for _ in range(draws)]
            return torch.stack(pred).mean(0)

def gaussian_kl(mu,lv): return (-.5*(1+lv-mu.square()-lv.exp()).sum(-1)).mean()

def band_loss(o,y,s, beta_z=.001,beta_u=.001,adv=.02,orth=.02,subject_effect=True,adversary=True,phys=.05):
    ce=F.cross_entropy(o['logits'],y)
    # KL[q(z_t|X) || p(z_t|z_(t-1),h_t)] along the temporal path.
    delta=o['step_mu']-o['prior_mu']
    klz=(.5*(o['prior_lv']-o['step_lv']+
       (o['step_lv'].exp()+delta.square())/o['prior_lv'].exp()-1)).sum(-1).mean()
    klu=gaussian_kl(o['um'],o['ul']) if subject_effect else ce.new_zeros(())
    ident=F.cross_entropy(o['id_logits'],s) if adversary else ce.new_zeros(())
    zc=o['z']-o['z'].mean(0,keepdim=True);uc=o['u']-o['u'].mean(0,keepdim=True)
    cov=(zc.T@uc)/max(1,len(s)-1)
    covpen=cov.square().mean() if subject_effect else ce.new_zeros(())
    phys_loss=F.smooth_l1_loss(o['beta_pred'],o['beta_target']) if phys else ce.new_zeros(())
    # Gradient reversal makes + CE train the discriminator while training the encoder to confuse it.
    return ce+beta_z*klz+beta_u*klu+adv*ident+orth*covpen+phys*phys_loss,{'ce':ce.item(),'kl_z':klz.item(),'kl_u':klu.item(),'identity_ce':ident.item(),'cov_penalty':covpen.item(),'phys_loss':phys_loss.item()}
