import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from genie.model.modules.invariant_point_attention import InvariantPointAttention
from genie.model.modules.structure_transition import StructureTransition
from genie.model.modules.backbone_update import BackboneUpdate


class StructureLayer(nn.Module):
    """
    Structure Layer.

    This module utilizes Invariant Point Attention (IPA) to combine multiple
    modes of representations (single, pair and frame) and subsequently update
    single representation. These update single representations are then used
    to compute backbone update.
    """

    def __init__(
        self,
        c_s,
        c_p,
        c_hidden,
        n_head,
        n_qk_point,
        n_v_point,
        ipa_dropout,
        n_structure_transition_layer,
        structure_transition_dropout
    ):
        """
        Args:
            c_s:
                Dimension of per-residue (single) representation.
            c_p:
                Dimension of paired residue-residue (pair) representation.
            c_hidden:
                Number of hidden dimensions in IPA layer.
            n_head:
                Number of heads in IPA layer.
            n_qk_point:
                Number of query/key points in IPA layer.
            n_v_point:
                Number of value points in IPA layer.
            ipa_dropout:
                Dropout rate in IPA layer.
            n_structure_transition_layer:
                Number of structure transition layers.
            structure_transition_dropout:
                Dropout rate in structure transition layer.
        """
        super(StructureLayer, self).__init__()

        # Invariant point attention
        self.ipa = InvariantPointAttention(
            c_s,
            c_p,
            c_hidden,
            n_head,
            n_qk_point,
            n_v_point
        )
        self.ipa_dropout = nn.Dropout(ipa_dropout)
        self.ipa_layer_norm = nn.LayerNorm(c_s)

        # Built-in dropout and layer norm
        self.transition = StructureTransition(
            c_s,
            n_structure_transition_layer,
            structure_transition_dropout
        )

        # Backbone update
        self.bb_update = BackboneUpdate(c_s)

    def forward(self, inputs):
        """
        Args:
            inputs:
                A tuple containing
                    s:
                        [B, N, c_s] single representation
                    p:
                        [B, N, N, c_p] pair representation
                    t:
                        [B, N] frames
                    mask:
                        [B, N] residue mask
                    states:
                        a running list to keep track of intermediate
                        single representations.

        Returns:
            outputs:
                A tuple containing
                    s:
                        [B, N, c_s] updated single representation
                    p:
                        [B, N, N, c_p] pair representation
                    t:
                        [B, N] frames
                    mask:
                        [B, N] residue mask
                    states:
                        a (updated) running list to keep track of
                        intermediate single representations.
        """
        s, p, t, mask, states = inputs
        s = s + self.ipa(s, p, t, mask)
        s = self.ipa_dropout(s)
        s = self.ipa_layer_norm(s)
        s = self.transition(s)
        states.append(s.unsqueeze(0))
        t = t.compose(self.bb_update(s))
        outputs = (s, p, t, mask, states)
        return outputs


class StructureNet(nn.Module):
    """
    Structure Network.

    This module utilizes multiple structure layers (with/without recycles)
    to compute and apply frame updates based on input single representations,
    pair representations and frames.

    Supports gradient checkpointing to reduce memory usage for large proteins.
    When enabled, activations are recomputed during backward pass instead of
    being stored, trading compute for memory (typically 4-10x memory reduction).
    """

    def __init__(
        self,
        c_s,
        c_p,
        n_structure_layer,
        n_structure_block,
        c_hidden_ipa,
        n_head_ipa,
        n_qk_point,
        n_v_point,
        ipa_dropout,
        n_structure_transition_layer,
        structure_transition_dropout
    ):
        """
        Args:
            c_s:
                Dimension of per-residue (single) representation.
            c_p:
                Dimension of paired residue-residue (pair) representation.
            n_structure_layer:
                Number of structure layers.
            n_structure_block:
                Number of recycles.
            c_hidden_ipa:
                Number of hidden dimensions in IPA layer.
            n_head_ipa:
                Number of heads in IPA layer.
            n_qk_point:
                Number of query/key points in IPA layer.
            n_v_point:
                Number of value points in IPA layer.
            ipa_dropout:
                Dropout rate in IPA layer.
            n_structure_transition_layer:
                Number of structure transition layers.
            structure_transition_dropout:
                Dropout rate in structure transition layer.
        """
        super(StructureNet, self).__init__()
        self.n_structure_block = n_structure_block
        self.n_structure_layer = n_structure_layer

        # Gradient checkpointing configuration
        # Set via set_checkpointing() method after model creation
        self.use_checkpointing = False
        self.blocks_per_ckpt = 1  # Number of layers per checkpoint block

        # Create structure layers using Sequential (maintains checkpoint compatibility)
        # Access layers via self.net[i] for checkpointing
        self.net = nn.Sequential(*[
            StructureLayer(
                c_s,
                c_p,
                c_hidden_ipa,
                n_head_ipa,
                n_qk_point,
                n_v_point,
                ipa_dropout,
                n_structure_transition_layer,
                structure_transition_dropout
            )
            for _ in range(n_structure_layer)
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

    def forward(self, s, p, ts, features):
        """
        Args:
            s:
                [B, N, c_s] Single representation.
            p:
                [B, N, N, c_p] Pair representation.
            ts:
                [B, N] Frames.
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
            A tuple containing
                states:
                    [1 + num_structure_block * num_structure_layer, B, N, c_s]
                    intermediate single representations
                ts:
                    [B, N] updated frames.

        """
        states = [s.unsqueeze(0)]
        mask = features['residue_mask']

        # Determine if we should use checkpointing
        # Only checkpoint when gradients are enabled (training/optimization)
        use_ckpt = self.use_checkpointing and torch.is_grad_enabled()

        for block_idx in range(self.n_structure_block):
            if use_ckpt:
                # Process layers with gradient checkpointing
                # Note: We checkpoint individual layers because each layer
                # modifies ts (frames) which has special structure
                for i in range(0, self.n_structure_layer, self.blocks_per_ckpt):
                    block_end = min(i + self.blocks_per_ckpt, self.n_structure_layer)
                    # Access layers from self.net (Sequential is indexable)
                    block_layers = [self.net[j] for j in range(i, block_end)]

                    # For checkpointing, we need to handle the states list carefully
                    # We checkpoint the core computation but manage states outside
                    for layer in block_layers:
                        # Checkpoint the layer forward pass
                        # We wrap only the expensive computation, not state management
                        def run_layer(s_in, p_in, t_rots, t_trans, mask_in, layer=layer):
                            # Reconstruct ts from components for checkpoint compatibility
                            from genie.utils.affine_utils import T
                            ts_in = T(t_rots, t_trans)
                            # Run layer without states (we manage states outside)
                            s_out = s_in + layer.ipa(s_in, p_in, ts_in, mask_in)
                            s_out = layer.ipa_dropout(s_out)
                            s_out = layer.ipa_layer_norm(s_out)
                            s_out = layer.transition(s_out)
                            ts_out = ts_in.compose(layer.bb_update(s_out))
                            return s_out, ts_out.rots, ts_out.trans

                        s, ts_rots, ts_trans = checkpoint(
                            run_layer, s, p, ts.rots, ts.trans, mask,
                            use_reentrant=False
                        )
                        # Reconstruct ts and append state
                        from genie.utils.affine_utils import T
                        ts = T(ts_rots, ts_trans)
                        states.append(s.unsqueeze(0))
            else:
                # Standard forward pass without checkpointing
                s, p, ts, mask, states = self.net((s, p, ts, mask, states))

        states = torch.concat(states, dim=0)
        return states, ts
