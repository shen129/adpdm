from adpdm import CplxUNet2d, Scheduler, Adpdm
import torch


unet = CplxUNet2d(
    embedding_type="positional",
    embedding_dim=128,
    embedding_norm="default"
)

scheduler = Scheduler(
    alpha=0.005,
    num_iters=200
)

adpdm = Adpdm(
    network=unet,
    scheduler=scheduler
).cuda()

x = torch.randn(1, 1, 256, 256, dtype=torch.cfloat, device="cuda")
t = torch.randint(0, 201, (1,), device="cuda")
print(unet(x, t).shape)
output = adpdm.denoise(x)
