import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from natsu.deltanet import gated_delta_recurrent, gated_delta_chunk
import torch.nn.functional as F

def test_chunk_matches_recurrent():
    torch.manual_seed(0)
    B,H,L,dk,dv = 2,3,77,16,8
    q = torch.randn(B,H,L,dk, dtype=torch.float64); k = F.normalize(torch.randn(B,H,L,dk, dtype=torch.float64), dim=-1)
    v = torch.randn(B,H,L,dv, dtype=torch.float64); g = -F.softplus(torch.randn(B,H,L, dtype=torch.float64))
    beta = torch.rand(B,H,L, dtype=torch.float64)
    o1,S1 = gated_delta_recurrent(q,k,v,g,beta)
    o2,S2 = gated_delta_chunk(q,k,v,g,beta,chunk=16)
    assert torch.allclose(o1,o2,atol=1e-8), (o1-o2).abs().max()
    assert torch.allclose(S1,S2,atol=1e-8)
    # state carry: split sequence
    oa,Sa = gated_delta_chunk(q[:,:,:40],k[:,:,:40],v[:,:,:40],g[:,:,:40],beta[:,:,:40],chunk=16)
    ob,Sb = gated_delta_recurrent(q[:,:,40:],k[:,:,40:],v[:,:,40:],g[:,:,40:],beta[:,:,40:],S0=Sa)
    assert torch.allclose(torch.cat([oa,ob],2),o1,atol=1e-8)

if __name__ == "__main__":
    test_chunk_matches_recurrent(); print("ok")
