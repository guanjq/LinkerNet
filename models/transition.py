import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.so3 import ApproxAngularDistribution, random_normal_so3, so3vec_to_rotation, \
    rotation_to_so3vec


def get_transitions(cfg_model, num_classes, num_bond_classes):
    trans_rot, trans_pos, trans_dis = None, None, None
    trans_link_pos, trans_link_cls, trans_link_bond = None, None, None
    cfg_diff = cfg_model.diffusion
    if cfg_model.train_frag_rot:
        trans_rot = RotationTransition(cfg_model.num_steps, var_sched_opt=cfg_diff.trans_rot_opt)
    if cfg_model.train_frag_pos:
        if cfg_model.frag_pos_prior == 'stat_distance':
            trans_dis = PositionTransition(cfg_model.num_steps, var_sched_opt=cfg_diff.trans_dis_opt)
        else:
            trans_pos = PositionTransition(cfg_model.num_steps, var_sched_opt=cfg_diff.trans_pos_opt)

    if cfg_model.train_link:
        trans_link_pos = PositionTransition(cfg_model.num_steps, var_sched_opt=cfg_diff.trans_link_pos_opt)
        trans_link_cls = FragmentCategoricalTransition(
            cfg_model.num_steps, num_classes, var_sched_opt=cfg_model.diffusion.trans_link_cls_opt)
    if cfg_model.train_bond:
        trans_link_bond = FragmentCategoricalTransition(
            cfg_model.num_steps, num_bond_classes, var_sched_opt=cfg_model.diffusion.trans_link_bond_opt)

    return trans_rot, trans_dis, trans_pos, trans_link_pos, trans_link_cls, trans_link_bond


def clampped_one_hot(x, num_classes):
    mask = (x >= 0) & (x < num_classes)  # (F, )
    x = x.clamp(min=0, max=num_classes-1)
    y = F.one_hot(x, num_classes) * mask[..., None]  # (F, K)
    return y


def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def advance_schedule(timesteps, scale_start, scale_end, width, return_alphas_bar=False):
    k = width
    A0 = scale_end
    A1 = scale_start

    a = (A0 - A1) / (sigmoid(-k) - sigmoid(k))
    b = 0.5 * (A0 + A1 - a)

    x = np.linspace(-1, 1, timesteps)
    y = a * sigmoid(- k * x) + b
    # print(y)

    alphas_cumprod = y
    alphas = np.zeros_like(alphas_cumprod)
    alphas[0] = alphas_cumprod[0]
    alphas[1:] = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    betas = np.clip(betas, 0, 1)
    if not return_alphas_bar:
        return betas
    else:
        return betas, alphas_cumprod


def segment_schedule(timesteps, time_segment, segment_diff):
    assert np.sum(time_segment) == timesteps
    alphas_cumprod = []
    for i in range(len(time_segment)):
        time_this = time_segment[i] + 1
        params = segment_diff[i]
        _, alphas_this = advance_schedule(time_this, **params, return_alphas_bar=True)
        alphas_cumprod.extend(alphas_this[1:])
    alphas_cumprod = np.array(alphas_cumprod)

    alphas = np.zeros_like(alphas_cumprod)
    alphas[0] = alphas_cumprod[0]
    alphas[1:] = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    betas = np.clip(betas, 0, 1)
    return betas


def get_beta_schedule(sche_type, num_timesteps, **kwargs):
    if sche_type == "quad":
        betas = (
            np.linspace(
                kwargs['beta_start'] ** 0.5,
                kwargs['beta_end'] ** 0.5,
                num_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif sche_type == "linear":
        betas = np.linspace(
            kwargs['beta_start'], kwargs['beta_end'], num_timesteps, dtype=np.float64
        )
    elif sche_type == "const":
        betas = kwargs['beta_end'] * np.ones(num_timesteps, dtype=np.float64)
    elif sche_type == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_timesteps, 1, num_timesteps, dtype=np.float64
        )
    elif sche_type == "sigmoid":
        s = dict.get(kwargs, 's', 6)
        betas = np.linspace(-s, s, num_timesteps)
        betas = sigmoid(betas) * (kwargs['beta_end'] - kwargs['beta_start']) + kwargs['beta_start']
    elif sche_type == "cosine":
        s = dict.get(kwargs, 's', 0.008)
        betas = cosine_beta_schedule(num_timesteps, s=s)
    elif sche_type == "advance":
        scale_start = dict.get(kwargs, 'scale_start', 0.999)
        scale_end = dict.get(kwargs, 'scale_end', 0.001)
        width = dict.get(kwargs, 'width', 2)
        betas = advance_schedule(num_timesteps, scale_start, scale_end, width)
    elif sche_type == 'segment':
        betas = segment_schedule(num_timesteps, kwargs['time_segment'], kwargs['segment_diff'])
    else:
        raise NotImplementedError(sche_type)
    assert betas.shape == (num_timesteps,)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


class VarianceSchedule(nn.Module):

    def __init__(self, num_steps=100, sche_type='cosine', s=0.01, **kwargs):
        super().__init__()
        T = num_steps
        if sche_type == 'cosine':
            t = torch.arange(0, num_steps + 1, dtype=torch.float)
            f_t = torch.cos((np.pi / 2) * ((t / T) + s) / (1 + s)) ** 2
            alpha_bars = f_t / f_t[0]
            betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
            betas = torch.cat([torch.zeros([1]), betas], dim=0)
            betas = betas.clamp_max(0.999)
            self.sche_repr = f'cos_{s}_n{num_steps}'
        else:
            betas = get_beta_schedule(
                sche_type=sche_type,
                num_timesteps=num_steps,
                **kwargs
            )
            betas = torch.from_numpy(betas).float()
            betas = torch.cat([torch.zeros([1]), betas], dim=0)
            alpha_bars = torch.cumprod(1 - betas, dim=0)
            self.sche_repr = f"{sche_type}_n{num_steps}"

        sigmas = torch.zeros_like(betas)
        for i in range(1, betas.size(0)):
            sigmas[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas = torch.sqrt(sigmas)

        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('alphas', 1 - betas)
        self.register_buffer('sigmas', sigmas)


class PositionTransition(nn.Module):

    def __init__(self, num_steps, var_sched_opt={}):
        super().__init__()
        self.num_steps = num_steps
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

    def add_noise(self, p_0, t, shift_center=False):
        """
        Compute q(xt | x0)
        Args:
            p_0:    (F, 3).
            t:  (F, ).
            shift_center: used in frag pos diffusion (we operate on the fragments zero CoM system)
        """
        alpha_bar = self.var_sched.alpha_bars[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1)

        e_rand = torch.randn_like(p_0)
        if shift_center:
            e_rand1, e_rand2 = e_rand[::2], e_rand[1::2]
            e_rand_center = ((e_rand1 + e_rand2) / 2).repeat_interleave(2, dim=0)
            e_rand -= e_rand_center

        p_noisy = c0 * p_0 + c1 * e_rand
        return p_noisy, e_rand

    def denoise(self, p_t, eps_p, t, shift_center=False):
        # Compute p(xt_1 | xt)
        # IMPORTANT:
        #   clampping alpha is to fix the instability issue at the first step (t=T)
        #   it seems like a problem with the ``improved ddpm''.
        alpha = self.var_sched.alphas[t].clamp_min(
            self.var_sched.alphas[-2]
        )
        alpha_bar = self.var_sched.alpha_bars[t]
        sigma = self.var_sched.sigmas[t].view(-1, 1)

        c0 = (1.0 / torch.sqrt(alpha + 1e-8)).view(-1, 1)
        c1 = ((1 - alpha) / torch.sqrt(1 - alpha_bar + 1e-8)).view(-1, 1)

        z = torch.where(
            (t > 1)[:, None].expand_as(p_t),
            torch.randn_like(p_t),
            torch.zeros_like(p_t),
        )
        if shift_center:
            z_rand1, z_rand2 = z[::2], z[1::2]
            z_rand_center = ((z_rand1 + z_rand2) / 2).repeat_interleave(2, dim=0)
            z -= z_rand_center
        p_next = c0 * (p_t - c1 * eps_p) + sigma * z
        return p_next

    def get_eps_from_p0_pt(self, p_0, p_t, t):
        alpha = self.var_sched.alphas[t].clamp_min(
            self.var_sched.alphas[-2]
        )
        alpha_bar = self.var_sched.alpha_bars[t]
        alpha_bar_prev = torch.where(t > 1, self.var_sched.alpha_bars[t - 1], torch.ones_like(alpha_bar))
        alpha_bar = alpha_bar.view(-1, 1)
        alpha_bar_prev = alpha_bar_prev.view(-1, 1)
        eps = (p_t - torch.sqrt(alpha_bar_prev + 1e-8) * p_0) / (torch.sqrt(1 - alpha_bar))
        return eps

    def posterior_sample(self, p_0, p_t, t, shift_center=False):
        alpha = self.var_sched.alphas[t].clamp_min(
            self.var_sched.alphas[-2]
        )
        alpha_bar = self.var_sched.alpha_bars[t]
        alpha_bar_pred = torch.where(t > 1, self.var_sched.alpha_bars[t-1], torch.ones_like(alpha_bar))
        beta = self.var_sched.betas[t]
        sigma = self.var_sched.sigmas[t].view(-1, 1)

        c0 = (beta * torch.sqrt(alpha_bar_pred + 1e-8) / (1. - alpha_bar)).view(-1, 1)
        ct = ((1. - alpha_bar_pred) * torch.sqrt(alpha + 1e-8) / (1. - alpha_bar)).view(-1, 1)

        z = torch.where(
            (t > 1)[:, None].expand_as(p_t),
            torch.randn_like(p_t),
            torch.zeros_like(p_t),
        )
        # z = z - z.mean(0)
        if shift_center:
            z_rand1, z_rand2 = z[::2], z[1::2]
            z_rand_center = ((z_rand1 + z_rand2) / 2).repeat_interleave(2, dim=0)
            z -= z_rand_center
        p_next = c0 * p_0 + ct * p_t + sigma * z
        return p_next


class RotationTransition(nn.Module):

    def __init__(self, num_steps, var_sched_opt={}, angular_distrib_fwd_opt={}, angular_distrib_inv_opt={},
                 cache_dir='.rot_trans'):
        super().__init__()
        self.num_steps = num_steps
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Forward (perturb)
        c1 = torch.sqrt(1 - self.var_sched.alpha_bars)  # (T,).
        cache_fwd_fn = os.path.join(cache_dir, self.var_sched.sche_repr + '.fwd_dist')
        if os.path.exists(cache_fwd_fn):
            print('load rotation transition fwd from ', cache_fwd_fn)
            self.angular_distrib_fwd = torch.load(cache_fwd_fn)
        else:
            print('computing the forward rotation transition..')
            self.angular_distrib_fwd = ApproxAngularDistribution(c1.tolist(), **angular_distrib_fwd_opt)
            torch.save(self.angular_distrib_fwd, cache_fwd_fn)
            print('save rotation transition fwd to ', cache_fwd_fn)

        # Inverse (generate)
        sigma = self.var_sched.sigmas
        cache_inv_fn = os.path.join(cache_dir, self.var_sched.sche_repr + '.inv_dist')
        if os.path.exists(cache_inv_fn):
            print('load rotation transition inv from ', cache_inv_fn)
            self.angular_distrib_inv = torch.load(cache_inv_fn)
        else:
            print('computing the inverse rotation transition..')
            self.angular_distrib_inv = ApproxAngularDistribution(sigma.tolist(), **angular_distrib_inv_opt)
            torch.save(self.angular_distrib_inv, cache_inv_fn)
            print('save rotation transition inv to ', cache_inv_fn)

        self.register_buffer('_dummy', torch.empty([0, ]))

    def add_noise(self, v_0, t):
        """
        Args:
            v_0:    (F, 3).
            t:  (F, ).
        """
        # N, L = mask_generate.size()
        alpha_bar = self.var_sched.alpha_bars[t]
        c0 = torch.sqrt(alpha_bar).view(-1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1)

        # Noise rotation
        e_scaled = random_normal_so3(t, self.angular_distrib_fwd, device=self._dummy.device)  # (F, 3)
        e_normal = e_scaled / (c1 + 1e-8)
        E_scaled = so3vec_to_rotation(e_scaled)  # (F, 3, 3)

        # Scaled true rotation
        R0_scaled = so3vec_to_rotation(c0 * v_0)  # (F, 3, 3)

        R_noisy = E_scaled @ R0_scaled
        v_noisy = rotation_to_so3vec(R_noisy)
        return v_noisy, e_normal

    def denoise(self, v_t, v_0_pred, t):
        """
        Compute p(xt_1 | xt)
        Args:
            v_0_pred: model predicted rotation
            t: (F, )
        """
        num_frags = t.size(0)
        # Scaled true rotation
        alpha_bar = self.var_sched.alpha_bars[t]
        alpha_bar_prev = torch.where(t > 1, self.var_sched.alpha_bars[t - 1], torch.ones_like(alpha_bar))
        alpha_bar = alpha_bar.view(-1, 1)
        alpha_bar_prev = alpha_bar_prev.view(-1, 1)
        betas = self.var_sched.betas[t].view(-1, 1)

        c0_post = (torch.sqrt(alpha_bar_prev) * betas / (1 - alpha_bar))
        ct_post = (torch.sqrt(1 - betas) * (1 - alpha_bar_prev) / (1 - alpha_bar))
        mu_R_next = so3vec_to_rotation(c0_post * v_0_pred) @ so3vec_to_rotation(ct_post * v_t)  # (F, 3, 3)

        # random rotation
        e = random_normal_so3(t, self.angular_distrib_inv, device=self._dummy.device)  # (F, 3)
        e = torch.where(
            (t > 1)[:, None].expand(num_frags, 3),
            e,
            torch.zeros_like(e)  # Simply denoise and don't add noise at the last step
        )
        E = so3vec_to_rotation(e)

        R_next = E @ mu_R_next
        v_next = rotation_to_so3vec(R_next)
        return v_next


class FragmentCategoricalTransition(nn.Module):

    def __init__(self, num_steps, num_classes, var_sched_opt={}):
        super().__init__()
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

    @staticmethod
    def _sample(c):
        """
        Args:
            c:    (F, K).
        Returns:
            x:    (F, ).
        """
        c = c + 1e-8
        x = torch.multinomial(c, 1).view(-1)
        return x

    def add_noise(self, x_0, t):
        """
        Args:
            x_0:    (F, )
            t:  (F, ).
        Returns:
            c_t:    Probability, (F, K).
            x_t:    Sample, LongTensor, (F, ).
        """
        K = self.num_classes
        c_0 = F.one_hot(x_0, num_classes=K).float()  # (F, K)
        alpha_bar = self.var_sched.alpha_bars[t][:, None]  # (F, 1)
        c_noisy = (alpha_bar * c_0) + ((1 - alpha_bar) / K)
        c_t = c_noisy
        x_t = self._sample(c_t)
        return c_t, x_t

    def posterior(self, x_t, x_0, t):
        """
        Args:
            x_t:    Category LongTensor (F, ) or Probability FloatTensor (F, K).
            x_0:    Category LongTensor (F, ) or Probability FloatTensor (F, K).
            t:  (N,).
        Returns:
            theta:  Posterior probability at (t-1)-th step, (N, L, K).
        """
        K = self.num_classes

        if x_t.dim() == 2:
            c_t = x_t  # When x_t is probability distribution.
        else:
            c_t = clampped_one_hot(x_t, num_classes=K).float()  # (F, K)

        if x_0.dim() == 2:
            c_0 = x_0  # When x_0 is probability distribution.
        else:
            c_0 = clampped_one_hot(x_0, num_classes=K).float()  # (F, K)

        alpha = self.var_sched.alpha_bars[t][:, None]  # (F, 1)
        alpha_bar = self.var_sched.alpha_bars[t][:, None]  # (F, 1)

        theta = ((alpha * c_t) + (1 - alpha) / K) * ((alpha_bar * c_0) + (1 - alpha_bar) / K)  # (F, K)
        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)
        return theta

    def denoise(self, x_t, c_0_pred, t):
        """
        Args:
            x_t:        (F, ).
            c_0_pred:   Normalized probability predicted by networks, (F, K).
            t:  (F, ).
        Returns:
            post:   Posterior probability at (t-1)-th step, (F, K).
            x_next: Sample at (t-1)-th step, LongTensor, (F, ).
        """
        c_t = clampped_one_hot(x_t, num_classes=self.num_classes).float()  # (F, K)
        post = self.posterior(c_t, c_0_pred, t=t)  # (F, K)
        x_next = self._sample(post)
        return post, x_next
