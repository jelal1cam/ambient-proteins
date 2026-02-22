import torch
from torch import nn
from torch.nn import functional as F
from itertools import groupby

from genie.utils.geo_utils import distance
from genie.utils.affine_utils import rot_to_quat

BINNING_KERNELS = ('hard', 'softmax', 'gaussian', 'colabdesign', 'colabdesign_ste')


class PairFeatureNet(nn.Module):
    """
    Pair Feature Network.

    This module generates paired residue-residue (pair) representations by 
    integrating the following information
        -   outer sum of the (single) residue representations
        -   relative positional encodings of residue pairs
        -   template of the input noised structure, including pairwise 
            distances and orientations between frames
        -   template of the motif structure, including pairwise distances 
            between motif Ca atoms (masked accordingly based on the 
            fixed structure mask in the feature dictionary).
    """

    def __init__(
        self,
        c_s,
        c_p,
        n_timestep,
        relpos_k,
        template_dist_min,
        template_dist_step,
        template_dist_n_bin
    ):
        """
        Args:
            c_s:
                Dimension of per-residue (single) representation.
            c_p:
                Dimension of paired residue-residue (pair) representation.
            n_timestep:
                Total number of diffusion timesteps.
            relpos_k:
                Window size used in relative positional encoding.
            template_dist_min:
                Minimum distance for pairwise distance bin.
            template_dist_step:
                Pairwise distance bin size.
            template_dist_n_bin:
                Number of pairwise distance bins.
        """
        super(PairFeatureNet, self).__init__()
        self.c_s = c_s
        self.c_p = c_p
        self.n_timestep = n_timestep

        # Binning configuration for differentiable optimisation
        # Options: 'hard' (default), 'softmax', 'gaussian', 'colabdesign', 'colabdesign_ste'
        self.binning_kernel = 'hard'
        self.binning_temperature = 1.0  # Higher = softer (standard convention)

        # Layers for outer sum of single representations
        self.linear_s_p_i = nn.Linear(c_s, c_p, bias=False)
        self.linear_s_p_j = nn.Linear(c_s, c_p, bias=False)

        # Parameters and layers for relative positional encoding
        self.relpos_k = relpos_k
        self.relpos_n_bin = 2 * relpos_k + 2
        self.linear_relpos = nn.Linear(self.relpos_n_bin + 1, c_p, bias=False)

        # Parameters and layers for templates
        self.template_dist_min = template_dist_min
        self.template_dist_step = template_dist_step
        self.template_dist_n_bin = template_dist_n_bin
        self.linear_template = nn.Linear(self.template_dist_n_bin + 6, c_p, bias=False)
        self.linear_motif_template = nn.Linear(self.template_dist_n_bin + 2, c_p, bias=False)

    def forward(self, s, ts, timesteps, features):
        """
        Args:
            s:
                [B, N, c_s] Per-residue (single) representation
            ts:
                [B, N] Frames at a given timestep
            timesteps:
                [B, N] Diffusion timestep
            features:
                A batched feature dictionary with a batch size B, where each 
                structure is padded to the maximum sequence length N. It contains 
                the following information
                    -   aatype: 
                            [B, N, 20] one-hot encoding on amino acid types
                    -   num_chains: 
                            [B, 1] number of chains in the structure
                    -   num_residues: 
                            [B, 1] number of residues in the structure
                    -   num_residues_per_chain: 
                            [B, 1] an array of number of residues by chain
                    -   atom_positions: 
                            [B, N, 3] an array of Ca atom positions
                    -   residue_mask: 
                            [B, N] residue mask to indicate which residue position is masked
                    -   residue_index: 
                            [B, N] residue index (started from 0)
                    -   chain_index: 
                            [B, N] chain index (started from 0)
                    -   fixed_sequence_mask: 
                            [B, N] mask to indicate which residue contains conditional
                            sequence information
                    -   fixed_structure_mask: 
                            [B, N, N] mask to indicate which pair of residues contains
                            conditional structural information
                    -   fixed_group:
                            [B, N] group index to indicate which group the residue belongs to
                            (useful for specifying multiple functional motifs)
                    -   interface_mask:
                            [B, N] deprecated and set to all zeros.

        Returns:
            [B, N, N, c_p] Paired residue-residue (pair) representation
        """

        # Pairwise residue masks
        # Shape: [B, N, N, 1]
        pair_residue_mask = features['residue_mask'].unsqueeze(1) * \
            features['residue_mask'].unsqueeze(2)

        # Linear projections of single representation
        # Shape: [B, N, c_p]
        p_i = self.linear_s_p_i(s)
        p_j = self.linear_s_p_j(s)

        # Outer sum of linear projections of single representation
        # Shape: [B, N, N, c_p]
        p = p_i[:, :, None, :] + p_j[:, None, :, :]

        # Aggregate pair representation with pairwise relative position 
        # encoding and template-based encodings
        # Shape: [B, N, N, c_p]
        p = p + self._relpos(features)
        p = p + self.linear_template(
            torch.cat([
                self._encode_positions(
                    ts.trans,
                    features['residue_mask']
                ),
                self._encode_orientations(
                    ts.rots,
                    features['residue_mask']
                ),
                features['fixed_structure_mask'].unsqueeze(-1),
                features['fixed_structure_mask'].unsqueeze(-1)
            ], axis=-1)
        )
        p = p + self.linear_motif_template(
            torch.cat([
                self._encode_positions(
                    features['atom_positions'],
                    features['fixed_sequence_mask']
                ) * features['fixed_structure_mask'].unsqueeze(-1),
                features['fixed_structure_mask'].unsqueeze(-1),
                features['fixed_structure_mask'].unsqueeze(-1)
            ], axis=-1)
        )

        return p * pair_residue_mask.unsqueeze(-1)

    ############################
    ###   Helper Functions   ###
    ############################

    def _relpos(self, features):
        """
        Compute relative position encoding based on residue indices
        (within the chain) and chain indices.

        This algorithm is adopted from AlphaFold 2 Algorithm 4 & 5
        and implemented based on OpenFold utils/tensor_utils.py.

        Args:
            features:
                A batched feature dictionary with a batch size B, where each 
                structure is padded to the maximum sequence length N. It contains 
                the following information that relates to this function
                    -   residue_mask: 
                            [B, N] residue mask to indicate which residue 
                            position is masked
                    -   residue_index: 
                            [B, N] residue index (started from 0)
                    -   chain_index: 
                            [B, N] chain index (started from 0)

        Returns:
            [B, N, N, c_p] Pair representation based on pairwise relative positions
        """
        residue_index = features['residue_index']
        chain_index = features['chain_index']
        residue_mask = features['residue_mask']

        # Same chain mask
        # Denotes if two residues are in the same chain
        # Shape: [B, N, N]
        is_same_chain = chain_index[:, :, None] == chain_index[:, None, :]

        # Pairwise relative position matrix, offsetted by window size
        # Note that relative residue position across chains is capped at
        # relpos_k + 1, or 2 * relpos_k + 1 with offset
        # Shape: [B, N, N]
        d_same_chain = torch.clip(
            residue_index[:, :, None] - residue_index[:, None, :] + self.relpos_k,
            0, 2 * self.relpos_k
        )
        d_diff_chain = torch.ones_like(d_same_chain) * (2 * self.relpos_k + 1)
        d = d_same_chain * is_same_chain + d_diff_chain * ~is_same_chain

        # Pairwise relative position encoding
        # Shape: [B, N, N, n_bin]
        oh = nn.functional.one_hot(d.long(), num_classes=self.relpos_n_bin).float()

        # Project to given single representation dimension
        # Shape: [B, N, N, c_p]
        return self.linear_relpos(torch.cat([
            oh,
            is_same_chain.unsqueeze(-1)
        ], axis=-1))

    def _encode_positions(self, coords, mask):
        """
        Encode pairwise distances for a sequence of coordinates.

        Supports five binning kernels (set via `self.binning_kernel`):
        - 'hard' (default): Uses argmin for exact bin assignment.
          Non-differentiable but matches pretrained model behaviour.
        - 'softmax': Uses softmax(-|d - c| / T) for differentiable binning.
          Has a V-shaped kink at the peak (non-smooth gradient).
        - 'gaussian': Uses softmax(-(d - c)^2 / T) for differentiable binning.
          Provides smoother gradients (parabolic peak), better for optimisation.
        - 'colabdesign': Uses sigmoid((d - lower)/T) * sigmoid((upper - d)/T).
          Community standard from ColabDesign/BindCraft. Proven in protein design.
        - 'colabdesign_ste': Straight-through estimator variant of colabdesign.
          Hard forward pass (exact bins), soft backward pass (gradients flow).

        Temperature convention (standard): higher T = softer distribution,
        lower T = sharper distribution (closer to hard binning).

        Args:
            coords:
                [B, N, 3] A sequence of atom positions.
            mask:
                [B, N] Mask to indicate which atom position is masked.

        Returns:
            [B, N, N, n_bin] Masked pairwise distance encoding.
        """

        # Pairwise distance matrix
        # Shape: [B, N, N]
        d = distance(torch.stack([
            coords.unsqueeze(2).repeat(1, 1, coords.shape[1], 1),
            coords.unsqueeze(1).repeat(1, coords.shape[1], 1, 1),
        ], dim=-2))

        # Distance bins
        # Shape: [n_bin]
        v = torch.arange(0, self.template_dist_n_bin, device=coords.device, dtype=coords.dtype)
        v = self.template_dist_min + v * self.template_dist_step

        # Reshaped distance bins
        # Shape: [1, 1, 1, n_bin]
        v_reshaped = v.view(*((1,) * len(d.shape) + (len(v),)))

        # Compute differences for binning
        # Shape: [B, N, N, n_bin]
        diffs = d.unsqueeze(-1) - v_reshaped

        if self.binning_kernel == 'hard':
            # ORIGINAL: hard binning (for inference/pretrained compatibility)
            # Non-differentiable argmin + one-hot
            # Shape: [B, N, N]
            b = torch.argmin(torch.abs(diffs), dim=-1)

            # Pairwise distance bin encoding (vmap-compatible, avoids scatter_)
            # Shape: [B, N, N, n_bin]
            oh = (b.unsqueeze(-1) == torch.arange(len(v), device=b.device)).float()

        elif self.binning_kernel == 'softmax':
            # DIFFERENTIABLE: softmax(-|d - c| / T)
            # Standard temperature convention: higher T = softer distribution
            # Lower T = sharper distribution (closer to hard binning)
            oh = F.softmax(-torch.abs(diffs) / self.binning_temperature, dim=-1)

        elif self.binning_kernel == 'gaussian':
            # DIFFERENTIABLE: softmax(-(d - c)^2 / T)
            # Gaussian kernel provides smoother gradients at the peak
            # Higher T = softer, Lower T = sharper
            oh = F.softmax(-diffs.pow(2) / self.binning_temperature, dim=-1)

        elif self.binning_kernel == 'colabdesign':
            # ColabDesign-style soft binning (community standard)
            # Uses product of sigmoids to create soft window for each bin
            # Reference: github.com/sokrypton/ColabDesign
            #
            # Each bin has lower and upper bounds computed from centres
            # sigmoid((d - lower)/T) * sigmoid((upper - d)/T) creates soft membership
            half_step = self.template_dist_step / 2
            lower_bounds = v_reshaped - half_step  # [1, 1, 1, n_bin]
            upper_bounds = v_reshaped + half_step  # [1, 1, 1, n_bin]

            # Edge handling: use -inf/+inf for first/last bins to capture all distances
            # This matches ColabDesign's approach for handling distances outside the bin range
            lower_bounds = lower_bounds.clone()
            upper_bounds = upper_bounds.clone()
            lower_bounds[..., 0] = -1e8
            upper_bounds[..., -1] = 1e8

            d_expanded = d.unsqueeze(-1)  # [B, N, N, 1]

            # Product of sigmoids creates soft window
            oh = (torch.sigmoid((d_expanded - lower_bounds) / self.binning_temperature) *
                  torch.sigmoid((upper_bounds - d_expanded) / self.binning_temperature))

            # Normalise to sum to 1
            oh = oh / (oh.sum(-1, keepdim=True) + 1e-8)

        elif self.binning_kernel == 'colabdesign_ste':
            # Straight-through estimator: hard forward, soft backward
            # Forward: use hard binning for exact pretrained compatibility
            b = torch.argmin(torch.abs(diffs), dim=-1)
            # vmap-compatible one_hot (avoids scatter_ which lacks batching rule)
            oh_hard = (b.unsqueeze(-1) == torch.arange(len(v), device=b.device)).float()

            # Backward: use ColabDesign soft binning for gradient flow
            half_step = self.template_dist_step / 2
            lower_bounds = v_reshaped - half_step
            upper_bounds = v_reshaped + half_step

            # Edge handling: use -inf/+inf for first/last bins to capture all distances
            lower_bounds = lower_bounds.clone()
            upper_bounds = upper_bounds.clone()
            lower_bounds[..., 0] = -1e8
            upper_bounds[..., -1] = 1e8

            d_expanded = d.unsqueeze(-1)
            oh_soft = (torch.sigmoid((d_expanded - lower_bounds) / self.binning_temperature) *
                       torch.sigmoid((upper_bounds - d_expanded) / self.binning_temperature))
            oh_soft = oh_soft / (oh_soft.sum(-1, keepdim=True) + 1e-8)

            # STE: hard values, soft gradients
            # Forward: oh_soft - oh_soft + oh_hard = oh_hard
            # Backward: gradients flow through oh_soft
            oh = oh_soft - oh_soft.detach() + oh_hard

        else:
            raise ValueError(f"Unknown binning kernel: '{self.binning_kernel}'. "
                           f"Valid options: {BINNING_KERNELS}")

        # Pairwise mask
        # Shape: [B, N, N]
        pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)

        return oh * pair_mask.unsqueeze(-1)

    def _encode_orientations(self, rots, mask):
        """
        Encode pairwise relative orientations for a sequence of frames.

        Args:
            rots:
                [B, N, 3, 3] A sequence of orientations.
            mask:
                [B, N] Mask to indicate which orientation is masked.

        Returns:
            [B, N, N, 4] Masked pairwise relative orientation encoding 
            (in terms of quaternions).
        """

        # Pairwise rotation matrix
        # Shape: [B, N, N, 3, 3]
        r = torch.matmul(
            rots.unsqueeze(1),
            rots.unsqueeze(2)
        )

        # Pairwise quaternion
        # Shape: [B, N, N, 4]
        q = rot_to_quat(r)

        # Pairwise mask
        # Shape: [B, N, N]
        pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)

        return q * pair_mask.unsqueeze(-1)  