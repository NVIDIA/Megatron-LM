import torch
from .hf_bagel_vae import load_ae
from typing import List, Tuple

from megatron.training import get_args



class DiffusionWrapper(torch.nn.Module):
    
    """Whisper audio encoder wrapper that extracts last_hidden_state."""
    def __init__(self):
        # Diffusion components need to be lazily loaded after megatron is initialized.
        # but we will put the instantiation before megatron training loop.
        super().__init__()
        self.vae = None
        self.vae_params = None
        self.latent_patch_size = None
        self.latent_downsample = None

    def init_vae(self, model_path: str, latent_patch_size: int, dtype: torch.dtype=torch.bfloat16):
        self.vae, self.vae_params = load_ae(model_path)
        self.vae.to(dtype).cuda()
        self.vae.eval()
        self.dtype = dtype
        self.latent_patch_size = latent_patch_size
        self.latent_downsample = self.vae_params.downsample * latent_patch_size

    def remove_vae(self):
        self.vae = None

    def cuda(self):
        if self.vae is not None:
            self.vae.cuda()
        self.device = torch.cuda.current_device()

    def set_timestep_shift(self, timestep_shift: float):
        self.timestep_shift = timestep_shift

    def vae_encode(self, padded_images: torch.Tensor, pachified_vae_latent_shapes: List[Tuple[int, int]]) -> List[torch.Tensor]:
        '''
        input_features: torch.Tensor
            input audio features
        seq_lengths: torch.Tensor
            the number of audio tokens corresponding to non-padded audio frames
            we only get the embeddings for the non-padded audio frames
        '''
        assert padded_images.ndim == 4, "padded_images should be a 4D tensor"
        assert len(pachified_vae_latent_shapes) == padded_images.shape[0], "the number of padded images should be equal to the number of pachified vae latent shapes"

        with torch.no_grad():
            # print(padded_images.dtype)
            # print(list(self.vae.parameters())[0].dtype)
            padded_latents = self.vae.encode(padded_images.to(self.dtype))

        packed_latents = []
        for latent, hw in zip(padded_latents, pachified_vae_latent_shapes):
            h, w = hw
            p = self.latent_patch_size
            c = self.vae_params.z_channels
            latent = latent[..., :h*p, :w*p].reshape(c, h, p, w, p)
            latent = torch.einsum('chpwq->hwpqc', latent).reshape(-1, c*p*p) #[s, c*p^2]
            packed_latents.append(latent)

        packed_latents_clean = torch.cat(packed_latents, dim=0)
            
        return packed_latents_clean #[packed_s,  c*p^2]

    def shift_timesteps(self, packed_timesteps: torch.Tensor) -> torch.Tensor:
        '''
        timesteps: torch.Tensor
        '''
        packed_timesteps = torch.sigmoid(packed_timesteps)
        packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
        return packed_timesteps

    def add_noise(self, packed_latents_clean: torch.Tensor, shifted_timesteps: torch.Tensor) -> torch.Tensor:
        '''
        packed_latents: torch.Tensor
        shifted_timesteps: torch.Tensor
        '''
        noise = torch.randn_like(packed_latents_clean)
        packed_latent = (1 - shifted_timesteps[:, None]) * packed_latents_clean + shifted_timesteps[:, None] * noise
        target = (noise - packed_latents_clean)[shifted_timesteps > 0]
        return packed_latent, noise, target


def build_diffusion_wrapper():
    args = get_args()
    vae_path = getattr(args, 'vae_path', None)
    if vae_path is None:
        return None
    # max_latent_size = args.max_latent_size # for gen pos embed
    latent_patch_size = args.latent_patch_size
    timestep_shift = args.timestep_shift
    # vae_cond_dropout_prob = args.vae_cond_dropout_prob # for gen dataset

    diffusion_wrapper = DiffusionWrapper()
    diffusion_wrapper.init_vae(vae_path, latent_patch_size)
    diffusion_wrapper.set_timestep_shift(timestep_shift)

    return diffusion_wrapper