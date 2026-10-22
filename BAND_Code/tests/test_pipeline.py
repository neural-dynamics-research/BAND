import numpy as np
import pandas as pd
import torch
from band.model import BAND,band_loss
from band.physiology import bandpower,burst_durations,participant_features
from band.clinical import paired_medication,clinical_association
from band.robustness import remove_channels
from band.eval import metrics

def test_model_sampling_and_gradient():
 torch.manual_seed(0); m=BAND(4,hidden=8,latent=4,n_subjects=4);x=torch.randn(8,4,128)
 output=m(x); loss,parts=band_loss(output,torch.tensor([0,1]*4),torch.tensor([0,1,2,3]*2))
 loss.backward();assert torch.isfinite(loss);assert m.predict_new(x[:2],draws=2).shape==(2,2)

def test_physio_bursts():
 rng=np.random.default_rng(0);x=rng.normal(size=(4,3,500))
 assert bandpower(x,250).shape==(4,3)
 assert len(burst_durations(x,250))==4
 assert len(participant_features(x,np.array(['a','a','b','b']),250))==2

def test_clinical_paired():
 data=pd.DataFrame({'subject_id':['a','a','b','b','c','c'],'medication':['OFF','ON']*3,'beta_power':[2.,1.,4.,3.,3.,2.]})
 paired,stats=paired_medication(data);assert stats['n_paired']==3;assert abs(stats['mean_ON_minus_OFF']+1)<1e-8
 clinical=pd.DataFrame({'subject_id':['a','b','c'],'mds_updrs_iii':[10,20,30],'beta_power':[1.,2.,3.]})
 _,fit=clinical_association(clinical);assert fit['n_participants']==3

def test_mask_and_metrics():
 x=np.ones((4,10,128));out,mask=remove_channels(x,.2,0)
 assert len(mask)==2 and (out[:,mask]==0).all()
 p=np.array([[.8,.2],[.2,.8],[.7,.3],[.1,.9]]);y=np.array([0,1,0,1]);assert metrics(y,p)['BA']==1


def test_auxiliary_phys_loss_is_active():
 torch.manual_seed(10)
 model=BAND(4,hidden=8,latent=4,n_subjects=3)
 x=torch.randn(6,4,250);y=torch.tensor([0,1,0,1,0,1]);sub=torch.tensor([0,1,2,0,1,2])
 out=model(x)
 loss_active,parts=band_loss(out,y,sub,phys=0.1)
 loss_none,_=band_loss(out,y,sub,phys=0.0)
 assert parts['phys_loss']>0 and loss_active.item()>loss_none.item()
 loss_active.backward()
 assert model.beta_head[-1].weight.grad is not None
