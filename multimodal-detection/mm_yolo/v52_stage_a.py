"""IR-only feature resampling: frozen coarse rotation + bounded learned residual.

No RGB feature or RGB-derived quality map enters the trainable module. The
coarse transform is explicitly RGB-referenced geometry from an offline cache.
Residual offsets are task-trained, NOT claims of recovered physical geometry.
"""
import torch
from torch import nn
from torch.nn import functional as F


def resample(x, sampling, canvas, delta=None):
    b, _, h, w = x.shape
    hc, wc = canvas
    yy, xx = torch.meshgrid(torch.arange(h, device=x.device, dtype=torch.float32),
                            torch.arange(w, device=x.device, dtype=torch.float32), indexing='ij')
    xy = torch.stack(((xx+.5)*wc/w-.5, (yy+.5)*hc/h-.5, torch.ones_like(xx)), -1)
    # Float coordinates remain accurate for large images under BF16 autocast.
    with torch.autocast(x.device.type, enabled=False):
        src = torch.einsum('bij,hwj->bhwi', sampling.float(), xy)
        if delta is not None:
            src = src + F.interpolate(delta.float(), (h,w), mode='bilinear', align_corners=False).permute(0,2,3,1)
        grid = torch.stack((2*(src[...,0]+.5)/wc-1, 2*(src[...,1]+.5)/hc-1), -1)
        out=F.grid_sample(x.float(), grid.float(), align_corners=False, padding_mode='zeros')
    return out.to(x.dtype)


class IRResidualInput(nn.Module):
    def __init__(self, p4_channels):
        super().__init__()
        self.offset = nn.Sequential(nn.Conv2d(p4_channels+3, 32, 3, padding=1),
                                    nn.GroupNorm(8,32), nn.SiLU(), nn.Conv2d(32,3,1))
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)
        self.raw_mix = nn.Parameter(torch.tensor(-2.2))
        self.last_penalty = None
        self.last_stats = {}

    def forward(self, raw, quality, canvas):
        if quality is None or 'v52_ir_sampling' not in quality:
            raise ValueError('V5.2 IR requires the versioned rotation cache at train AND eval time')
        sampling = quality['v52_ir_sampling']
        valid=quality['availability'][:,1:2]
        own=quality['ir'][:,[1,5,8]]  # contrast, blur, thermal proxy; IR-derived only
        p4=raw['p4']; shape=p4.shape[-2:]
        mask=F.interpolate(valid.float(), shape, mode='nearest').to(p4.dtype)
        coarse=resample(p4*mask,sampling,canvas)
        q=resample(F.interpolate(own.float(),shape,mode='bilinear',align_corners=False), sampling,canvas)
        pred=self.offset(torch.cat((coarse,q.to(coarse.dtype)),1))
        unit=pred[:,:2].tanh()
        gate=pred[:,2:3].sigmoid()
        # Up to 8% of each canvas dimension, spatially varying; starts at zero.
        bounds=unit.new_tensor([canvas[1]*.08,canvas[0]*.08])[None,:,None,None]
        delta=unit*bounds
        smooth=(unit[:,:,1:]-unit[:,:,:-1]).square().mean()+(unit[:,:,:,1:]-unit[:,:,:,:-1]).square().mean()
        self.last_penalty=unit.square().mean()+.2*smooth
        result={}
        for s,x in raw.items():
            v=F.interpolate(valid.float(),x.shape[-2:],mode='nearest').to(x.dtype)
            base=resample(x*v,sampling,canvas)
            refined=resample(x*v,sampling,canvas,delta)
            g=F.interpolate(gate,x.shape[-2:],mode='bilinear',align_corners=False)
            aligned=base+g*(refined-base)
            result[s]=aligned+self.raw_mix.sigmoid()*(x-aligned)
        self.last_stats={'flow_rms_px':float(delta.detach().float().square().mean().sqrt()),
                         'raw_mix':float(self.raw_mix.detach().sigmoid()),
                         'gate_mean':float(gate.detach().float().mean())}
        return result
