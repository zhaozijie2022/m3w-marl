import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math
import einops

# region MLP-based modules
class SimNorm(nn.Module):
    def __init__(self, simnorm_dim):
        super().__init__()
        self.dim = simnorm_dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"

class NormedLinear(nn.Linear):
    """
        Linear layer with LayerNorm, activation, and optionally dropout.
    """

    def __init__(self, *args, dropout=0., act=nn.Mish(inplace=True), **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return f"NormedLinear(in_features={self.in_features}, " \
               f"out_features={self.out_features}, " \
               f"bias={self.bias is not None}{repr_dropout}, " \
               f"act={self.act.__class__.__name__})"

class ActedLinear(nn.Linear):
    """
        Linear layer with activation
    """

    def __init__(self, *args, act=nn.Mish(inplace=True), **kwargs):
        super().__init__(*args, **kwargs)
        self.act = act

    def forward(self, x):
        x = super().forward(x)
        return self.act(x)

    def __repr__(self):
        return f"ActedLinear(in_features={self.in_features}, " \
               f"out_features={self.out_features}, " \
               f"bias={self.bias is not None}, " \
               f"act={self.act.__class__.__name__})"

def create_mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0., device='cpu', normed=True):
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    if normed:
        for i in range(len(dims) - 2):
            mlp.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0)))
        mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
    else:
        for i in range(len(dims) - 2):
            mlp.append(ActedLinear(dims[i], dims[i + 1], act=act, dropout=dropout * (i == 0)))
        mlp.append(nn.Linear(dims[-2], dims[-1]))

    return nn.Sequential(*mlp).to(device)

class MLP(nn.Module):
    def __init__(self, in_dim, mlp_dims, out_dim, act=None, dropout=0., lr=1e-3, device='cpu'):
        super().__init__()
        self.mlp = create_mlp(in_dim, mlp_dims, out_dim, act, dropout, device)
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr
        )
        self.to(device)

    def forward(self, x):
        return self.mlp(x)

    def turn_on_grad(self):
        for param in self.mlp.parameters():
            param.requires_grad = True

    def turn_off_grad(self):
        for param in self.mlp.parameters():
            param.requires_grad = False

    def save(self, **kwargs):
        assert "save_dir" in kwargs
        assert "model_name" in kwargs
        if "agent_id" in kwargs:
            torch.save(
                self.mlp.state_dict(), str(kwargs["save_dir"]) + '/' + kwargs["model_name"] + str(kwargs["agent_id"]) + ".pt"
            )
        else:
            torch.save(
                self.mlp.state_dict(), str(kwargs["save_dir"]) + '/' + kwargs["model_name"] + ".pt"
            )

    def restore(self, **kwargs):
        assert "load_dir" in kwargs
        assert "model_name" in kwargs
        if "agent_id" in kwargs:
            self.mlp.load_state_dict(
                torch.load(str(kwargs["load_dir"]) + '/' + kwargs["model_name"] + str(kwargs["agent_id"]) + ".pt")
            )
        else:
            self.mlp.load_state_dict(
                torch.load(str(kwargs["load_dir"]) + '/' + kwargs["model_name"] + ".pt")
            )

class MLPEncoder(MLP):
    def __init__(self, in_dim, mlp_dims, out_dim, act=None, dropout=0., lr=1e-3, device='cpu'):
        super().__init__(in_dim, mlp_dims, out_dim, act, dropout, lr, device)

    def encode(self, x):
        """
        Args:
            x:  (batch_size, dim)
        Returns:
            latent: (batch_size, latent_dim)
        """
        return self.forward(x)

class DecMLPPredictor(MLP):
    def __init__(self, in_dim, mlp_dims, out_dim, act=None, dropout=0., lr=1e-3, device='cpu'):
        super().__init__(in_dim, mlp_dims, out_dim, act, dropout, lr, device)

    def predict(self, z, a):
        """
        Args:
            z: (batch_size, latent_dim)
            a: (batch_size, action_dim_sum)
        Returns:
            next_state_latent_pred: (batch_size, latent_dim)
            reward_pred_logits: (batch_size, num_bins)
        """
        x = torch.cat([z, a], dim=-1)
        return self.forward(x)

class CenMLPDynamicsModel(MLP):
    def __init__(self, in_dim, mlp_dims, out_dim, act=None, dropout=0., lr=1e-3, device='cpu'):
        super().__init__(in_dim, mlp_dims, out_dim, act, dropout, lr, device)

    def predict(self, z, a):
        """
        Args:
            z: (batch_size, num_agents, latent_dim)
            a: (batch_size, num_agents, action_dim)
        Returns:
            next_state_latent_pred: (batch_size, latent_dim)
            reward_pred_logits: (batch_size, num_bins)
        """
        batch_size, num_agents, d_model = z.shape
        z = z.reshape(batch_size, -1)
        a = a.reshape(batch_size, -1)
        x = torch.cat([z, a], dim=-1)
        y = self.forward(x)
        y = y.reshape(batch_size, num_agents, -1)
        return y
    

class CenMLPRewardModel(MLP):
    def __init__(self, in_dim, mlp_dims, out_dim, act=None, dropout=0., lr=1e-3, device='cpu'):
        super().__init__(in_dim, mlp_dims, out_dim, act, dropout, lr, device)

    def predict(self, z, a):
        batch_size, _, _ = z.shape
        z = z.reshape(batch_size, -1)
        a = a.reshape(batch_size, -1)
        x = torch.cat([z, a], dim=-1)
        y = self.forward(x)
        return y
# endregion


# region MoE-based modules
class SelfAttnExpert(nn.Module):
    def __init__(self, d_model, n_heads=4, ffn_hidden=2048, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, N_a, D]
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x  # [B, N_a, D]

class NoisyTopKRouter(nn.Module):
    def __init__(self, in_dim, num_experts, k=2, noisy_gating=True, device="cpu"):
        super().__init__()
        assert k <= num_experts
        self.in_dim = in_dim
        self.num_experts = num_experts
        self.k = k
        self.noisy_gating = noisy_gating

        self.w_gate = nn.Parameter(torch.zeros(in_dim, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(in_dim, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.to(device)

    @staticmethod
    def cv_squared(x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    @staticmethod
    def z_loss(logits):
        return torch.log(torch.exp(logits).sum(-1)).mean()

    @staticmethod
    def _gates_to_load(gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def forward(self, x_flat):
        """
        x_flat: [B, in_dim]  <-([B, N_a, D])
        return:
          gates:  [B, N_e]
          load:   [N_e]
          logits: [B, N_e]
          aux:    dict(loss_balancing=..., importance=..., load=...)
        """
        clean_logits = x_flat @ self.w_gate  # [B, N_e]
        if self.noisy_gating and self.training:
            raw_noise_stddev = x_flat @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + 1e-2
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            noise_stddev = None
            noisy_logits = None
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]          # [B, K]
        top_k_indices = top_indices[:, :self.k]        # [B, K]
        top_k_gates = self.softmax(top_k_logits)       # [B, K]

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)  # [B, N_e]

        if self.noisy_gating and self.k < self.num_experts and self.training:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)

        importance = gates.sum(0)
        load_balancing = self.cv_squared(importance) + self.cv_squared(load) + self.z_loss(logits)

        aux = dict(
            loss_balancing=load_balancing,
            importance=importance,
            load=load,
            logits=logits
        )
        return gates, load, logits, aux

class CenMoERewardModel(nn.Module):
    """
    input:
        a: [B, N_a, d_a]
        z: [B, N_a, d_z]
    process:
        1) x = concat(a, z) -> [B, N_a, D]  (D = d_a + d_z)
        2) Router(x_flat) -> gates, select top-K experts
        3) Self-Attn Experts forward on selected batch subset
        4) Weighted aggregate expert outputs
        5) Reward head per agent
    """
    def __init__(self, d_z, d_a, n_agents, n_experts, k, n_r, n_heads=1,
                 expert_ffn_hidden=1024, expert_dropout=0.0, head_hidden=512,
                 noisy_gating=True, device="cpu", lr=1e-3):
        super().__init__()
        self.D = d_z + d_a
        self.n_agents = n_agents
        self.n_experts = n_experts
        self.k = k
        self.n_r = n_r

        self.router = NoisyTopKRouter(
            in_dim=n_agents * self.D,
            num_experts=n_experts,
            k=k,
            noisy_gating=noisy_gating,
            device=device
        )

        self.experts = nn.ModuleList([
            SelfAttnExpert(d_model=self.D, n_heads=n_heads,
                           ffn_hidden=expert_ffn_hidden, dropout=expert_dropout)
            for _ in range(n_experts)
        ])
        
        self.reward_head = nn.Sequential(
            nn.Linear(n_agents * self.D, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, n_r)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.to(device)

    def forward(self, z, a):
        """
        a: [B, N_a, d_a]
        z: [B, N_a, d_z]
        return:
            r_logits: [B, N_a, N_r]
            aux: dict(loss_balancing=..., gates=..., logits=...)
        """
        B, N_a, d_a = a.shape
        x = torch.cat([z, a], dim=-1)
    
        x_flat = x.reshape(B, -1)                      # [B, N_a*D]
        gates, load, logits, aux_router = self.router(x_flat)  # gates: [B, N_e]

        y = torch.zeros_like(x)                        # [B, N_a, D]
        for e_idx, expert in enumerate(self.experts):
            mask = gates[:, e_idx] > 0
            if not mask.any():
                continue
            x_sel = x[mask]
            out_e = expert(x_sel)  # [b_i, N_a, D]
            w = gates[mask, e_idx].view(-1, 1, 1)
            y[mask] = y[mask] + w * out_e          #
        
        y_flat = y.reshape(B, -1)
        r_logits = self.reward_head(y_flat)                # [B, N_a, N_r]

        aux = dict(
            loss_balancing=aux_router["loss_balancing"],
            gates=gates,
            logits=aux_router["logits"]
        )
        return r_logits, aux
    
    def predict(self, z, a):
        y, aux = self.forward(z, a)
        return y
    
    def turn_on_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    def turn_off_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def save(self, **kwargs):
        assert "save_dir" in kwargs
        assert "model_name" in kwargs
        if "agent_id" in kwargs:
            path = str(kwargs["save_dir"]) + '/' + kwargs["model_name"] + str(kwargs["agent_id"]) + ".pt"
        else:
            path = str(kwargs["save_dir"]) + '/' + kwargs["model_name"] + ".pt"
        torch.save(self.state_dict(), path)

    def restore(self, **kwargs):
        assert "load_dir" in kwargs
        assert "model_name" in kwargs
        if "agent_id" in kwargs:
            path = str(kwargs["load_dir"]) + '/' + kwargs["model_name"] + str(kwargs["agent_id"]) + ".pt"
        else:
            path = str(kwargs["load_dir"]) + '/' + kwargs["model_name"] + ".pt"
        self.load_state_dict(torch.load(path))
        
class CenMoEDynamicsModel(nn.Module):
    def __init__(self, d_z, d_a, mlp_dims, n_experts, act=None, dropout=0.0, device="cpu", lr=1e-3):
        super().__init__()
        self.d_z = d_z
        self.d_a = d_a
        self.d_model = d_z + d_a
        self.n_experts = n_experts
        
        self.experts = nn.ModuleList([
            create_mlp(
                in_dim=d_z + d_a,
                mlp_dims=mlp_dims,
                out_dim=d_z,
                act=act,
                dropout=dropout,
                device=device,
            )
            for _ in range(n_experts)
        ])
        self.phi = nn.Parameter(
            torch.randn(d_z+d_a, n_experts, 1) * (1 / math.sqrt(d_z + d_a)),
        ).to(device)
    
    def predict(self, z, a):
        """
        z: [B, N_a, d_z]
        a: [B, N_a, d_a]
        """
        x = torch.cat([z, a], dim=-1)  # [B, N_a, d_model]
        
        # router weights: [B, N_a, N_e, S]
        weights = torch.einsum("b n d , d e s -> b n e s", x, self.phi)
        
        # dispatch: token -> experts
        dispatch_weights = F.softmax(weights, dim=1)  # [B, N_a, N_e, S]
        experts_inputs = torch.einsum("b n e s, b n d -> b e s d", dispatch_weights, x)
        # -> [B, N_e, S, d_model]
        
        # expert forward
        expert_outputs = torch.stack([
            self.experts[i](experts_inputs[:, i]) for i in range(self.n_experts)
        ])  # [N_e, B, S, d_z]
        expert_outputs = einops.rearrange(expert_outputs, "e b s d -> b (e s) d")
        
        # combine
        combine_weights = einops.rearrange(weights, "b n e s -> b n (e s)")
        combine_weights = F.softmax(combine_weights, dim=-1)
        out = torch.einsum("b n z, b z d -> b n d", combine_weights, expert_outputs)
        # [B, N_a, d_z]
        
        return out
    
    def turn_on_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    def turn_off_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def save(self, **kwargs):
        assert "save_dir" in kwargs
        assert "model_name" in kwargs
        if "agent_id" in kwargs:
            path = str(kwargs["save_dir"]) + '/' + kwargs["model_name"] + str(kwargs["agent_id"]) + ".pt"
        else:
            path = str(kwargs["save_dir"]) + '/' + kwargs["model_name"] + ".pt"
        torch.save(self.state_dict(), path)

    def restore(self, **kwargs):
        assert "load_dir" in kwargs
        assert "model_name" in kwargs
        if "agent_id" in kwargs:
            path = str(kwargs["load_dir"]) + '/' + kwargs["model_name"] + str(kwargs["agent_id"]) + ".pt"
        else:
            path = str(kwargs["load_dir"]) + '/' + kwargs["model_name"] + ".pt"
        self.load_state_dict(torch.load(path))
        
# endregion

# region Processing modules
class TwoHotProcessor:
    def __init__(self, num_bins, vmin, vmax, device):
        self.num_bins = num_bins
        self.vmin, self.vmax = vmin, vmax
        if num_bins > 1:
            self.bin_size = (vmax - vmin) / (num_bins - 1)
            self.dis_reg_bins = torch.linspace(vmin, vmax, num_bins, device=device)
        else:
            self.bin_size = 0.0
            self.dis_reg_bins = None

        self.sym_log = lambda x: torch.sign(x) * torch.log(1 + torch.abs(x))
        self.sym_exp = lambda x: torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

    def logits_decode_scalar(self, x):
        """logits -> scalars：[*, num_bins] → [*, 1]）"""
        if self.num_bins == 0:
            return x
        elif self.num_bins == 1:
            return self.sym_exp(x)
        else:
            x_softmax = F.softmax(x, dim=-1)
            weighted_sum = torch.sum(x_softmax * self.dis_reg_bins, dim=-1, keepdim=True)
            return self.sym_exp(weighted_sum)

    def scalar_encode_logits(self, x):
        """scalars -> two-hot [batch_size, 1] -> [batch_size, num_bins]"""
        if self.num_bins == 0:
            return x
        elif self.num_bins == 1:
            return self.sym_log(x)
        else:
            x_sym_log = self.sym_log(x)
            x_clamped = torch.clamp(x_sym_log, self.vmin, self.vmax).squeeze(1)

            bin_idx = torch.floor((x_clamped - self.vmin) / self.bin_size).long()
            bin_offset = ((x_clamped - self.vmin) / self.bin_size - bin_idx.float()).unsqueeze(-1)

            soft_two_hot = torch.zeros(x.size(0), self.num_bins, device=x.device)
            next_bin = (bin_idx + 1) % self.num_bins

            soft_two_hot.scatter_(1, bin_idx.unsqueeze(1), 1 - bin_offset)
            soft_two_hot.scatter_(1, next_bin.unsqueeze(1), bin_offset)
            return soft_two_hot

    def dis_reg_loss(self, logits, target):
        if self.num_bins == 0:
            return F.mse_loss(logits, target)
        elif self.num_bins == 1:
            return F.mse_loss(self.logits_decode_scalar(logits), target)
        else:
            log_pred = F.log_softmax(logits, dim=-1)
            target = self.scalar_encode_logits(target)
            return -(target * log_pred).sum(dim=-1, keepdim=True)


class RunningScale:
    def __init__(self, tpdv, tau):
        self._value = torch.ones(1).to(**tpdv)
        self._percentiles = torch.tensor([5, 95]).to(**tpdv)
        self.tau = tau

    @property
    def value(self):
        return self._value.cpu().item()

    def _percentile(self, x):
        x_dtype, x_shape = x.dtype, x.shape
        x = x.view(x.shape[0], -1)
        in_sorted, _ = torch.sort(x, dim=0)
        positions = self._percentiles * (x.shape[0] - 1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > x.shape[0] - 1] = x.shape[0] - 1
        weight_ceiled = positions - floored
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
        return (d0 + d1).view(-1, *x_shape[1:]).type(x_dtype)

    def update(self, x):
        percentiles = self._percentile(x.detach())
        value = torch.clamp(percentiles[1] - percentiles[0], min=1.)
        self._value.data.lerp_(value, self.tau)

    def state_dict(self):
        return dict(value=self._value, percentiles=self._percentiles)

    def load_state_dict(self, state_dict):
        self._value.data.copy_(state_dict['value'])
        self._percentiles.data.copy_(state_dict['percentiles'])

    def __call__(self, x, update=False):
        if update:
            self.update(x)
        return x * (1 / self.value)

    def __repr__(self):
        return f'RunningScale(S: {self.value})'