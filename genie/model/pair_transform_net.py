import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from genie.model.modules.pair_transition import PairTransition
from genie.model.modules.triangular_attention import (
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode,
)
from genie.model.modules.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)
from genie.model.modules.dropout import (
    DropoutRowwise,
    DropoutColumnwise
)


class PairTransformLayer(nn.Module):
    """
    Pair Transform Layer.

    Adapted from Evoformer, this module utilizes a triangular multiplicative
    update layer and a triangular attention layer (if specified) to refine
    pair representation.
    """

    def __init__(
        self,
        c_p,
        include_mul_update,
        include_tri_att,
        c_hidden_mul,
        c_hidden_tri_att,
        n_head_tri,
        tri_dropout,
        pair_transition_n,
    ):
        """
        Args:
            c_p:
                Dimension of paired residue-residue (pair) representation.
            include_mul_update:
                Flag on whether to use triangular multiplicative update layer.
            include_tri_att:
                Flag on whether to use triangular attention layer.
            c_hidden_mul:
                Number of hidden dimensions in triangular multiplicative update layer.
            c_hidden_tri_att:
                Number of hidden dimensions in triangular attention layer.
            n_head_tri:
                Number of heads in triangular attention layer.
            tri_dropout:
                Dropout rate.
            pair_transition_n:
                Number of pair transition layers.
        """
        super(PairTransformLayer, self).__init__()

        # Layers for triangular multiplicative updates
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_p,
            c_hidden_mul
        ) if include_mul_update else None
        self.tri_mul_in = TriangleMultiplicationIncoming(
            c_p,
            c_hidden_mul
        ) if include_mul_update else None

        # Layers for triangular attention
        self.tri_att_start = TriangleAttentionStartingNode(
            c_p,
            c_hidden_tri_att,
            n_head_tri
        ) if include_tri_att else None
        self.tri_att_end = TriangleAttentionEndingNode(
            c_p,
            c_hidden_tri_att,
            n_head_tri
        ) if include_tri_att else None

        # Layer for pair transition
        self.pair_transition = PairTransition(
            c_p,
            pair_transition_n
        )

        # Layers for dropouts
        self.dropout_row_layer = DropoutRowwise(tri_dropout)
        self.dropout_col_layer = DropoutColumnwise(tri_dropout)

    def forward(self, inputs):
        """
        Args:
            inputs:
                A tuple containing
                    p:
                        [B, N, N, c_p] pair representation
                    pair_residue_mask:
                        [B, N, N] pairwise residue mask.

        Returns:
            outputs:
                A tuple containing
                    p:
                        [B, N, N, c_p] updated pair representation
                    pair_residue_mask:
                        [B, N, N] pairwise residue mask.
        """
        p, pair_residue_mask = inputs
        if self.tri_mul_out is not None:
            p = p + self.dropout_row_layer(self.tri_mul_out(p, pair_residue_mask))
            p = p + self.dropout_row_layer(self.tri_mul_in(p, pair_residue_mask))
        if self.tri_att_start is not None:
            p = p + self.dropout_row_layer(self.tri_att_start(p, pair_residue_mask))
            p = p + self.dropout_col_layer(self.tri_att_end(p, pair_residue_mask))
        p = p + self.pair_transition(p, pair_residue_mask)
        p = p * pair_residue_mask.unsqueeze(-1)
        outputs = (p, pair_residue_mask)
        return outputs

class PairTransformNet(nn.Module):
    """
    Pair Transform Network.

    Adapted from Evoformer, this module utilizes multiple pair transform
    layers to refine pair representations before using them in the
    structure module.

    Supports gradient checkpointing to reduce memory usage for large proteins.
    When enabled, activations are recomputed during backward pass instead of
    being stored, trading compute for memory (typically 4-10x memory reduction).
    """

    def __init__(
        self,
        c_p,
        n_pair_transform_layer,
        include_mul_update,
        include_tri_att,
        c_hidden_mul,
        c_hidden_tri_att,
        n_head_tri,
        tri_dropout,
        pair_transition_n
    ):
        """
        Args:
            c_p:
                Dimension of paired residue-residue (pair) representation.
            n_pair_transform_layer:
                Number of pair transform layers.
            include_mul_update:
                Flag on whether to use triangular multiplicative update layer.
            include_tri_att:
                Flag on whether to use triangular attention layer.
            c_hidden_mul:
                Number of hidden dimensions in triangular multiplicative update layer.
            c_hidden_tri_att:
                Number of hidden dimensions in triangular attention layer.
            n_head_tri:
                Number of heads in triangular attention layer.
            tri_dropout:
                Dropout rate.
            pair_transition_n:
                Number of pair transition layers.
        """
        super(PairTransformNet, self).__init__()

        # Gradient checkpointing configuration
        # Set via set_checkpointing() method after model creation
        self.use_checkpointing = False
        self.blocks_per_ckpt = 1  # Number of layers per checkpoint block

        # Create pair transform layers using Sequential (maintains checkpoint compatibility)
        # Access layers via self.net[i] for checkpointing
        self.net = nn.Sequential(*[
            PairTransformLayer(
                c_p,
                include_mul_update,
                include_tri_att,
                c_hidden_mul,
                c_hidden_tri_att,
                n_head_tri,
                tri_dropout,
                pair_transition_n
            )
            for _ in range(n_pair_transform_layer)
        ])

    def set_checkpointing(self, enabled=True, blocks_per_ckpt=1):
        """Enable or disable gradient checkpointing.

        Args:
            enabled: Whether to use gradient checkpointing.
            blocks_per_ckpt: Number of layers to group per checkpoint.
                Higher values = fewer checkpoints = less memory savings but faster.
                Recommended: 1 for maximum memory savings, 2-3 for balance.
        """
        self.use_checkpointing = enabled
        self.blocks_per_ckpt = blocks_per_ckpt

    def forward(self, p, features):
        """
        Args:
            p:
                [B, N, N, c_p] Pair representation.
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
            [B, N, N, c_p] Updated pair representation.
        """

        # Pairwise residue mask
        # Shape: [B, N, N]
        pair_residue_mask = features['residue_mask'].unsqueeze(1) * features['residue_mask'].unsqueeze(2)

        # Determine if we should use checkpointing
        # Only checkpoint when gradients are enabled (training/optimization)
        use_ckpt = self.use_checkpointing and torch.is_grad_enabled()

        if use_ckpt:
            # Process layers with gradient checkpointing
            # Group layers into blocks for efficiency
            n_layers = len(self.net)
            for i in range(0, n_layers, self.blocks_per_ckpt):
                block_end = min(i + self.blocks_per_ckpt, n_layers)
                # Access layers from self.net (Sequential is indexable)
                block_layers = [self.net[j] for j in range(i, block_end)]

                # Define block function for checkpointing
                def run_block(p_in, mask_in, layers=block_layers):
                    p_out, mask_out = p_in, mask_in
                    for layer in layers:
                        p_out, mask_out = layer((p_out, mask_out))
                    return p_out, mask_out

                # Checkpoint the block
                # use_reentrant=False is recommended for newer PyTorch versions
                p, pair_residue_mask = checkpoint(
                    run_block, p, pair_residue_mask,
                    use_reentrant=False
                )
        else:
            # Standard forward pass without checkpointing
            p, _ = self.net((p, pair_residue_mask))

        return p
