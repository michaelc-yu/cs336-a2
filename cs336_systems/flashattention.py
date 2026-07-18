import torch
import math
from einops import einsum, rearrange


# Tile sizes
B_Q = 16
B_K = 16

class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Inputs: Q, K, V
        Outputs: O, L
        """
        # For one batch:
        # Q has shape Nq x d
        # K has shape Nk x d
        # V has shape Nk x d
        assert K.shape[1] == V.shape[1]

        batch_sz, N_q, d = Q.shape
        _, N_k , _ = K.shape
        
        assert N_q % B_Q == 0 # num rows in Q is divisible by tile size
        assert N_k % B_K == 0 # num rows in K,V is divisible by tile size

        T_q = math.ceil(N_q / B_Q)
        T_k = math.ceil(N_k / B_K)

        device = Q.device
        dtype = Q.dtype

        O = torch.zeros((batch_sz, N_q, d), device=device, dtype=dtype)
        L = torch.zeros((batch_sz, N_q), device=device, dtype=dtype) # logsumexp

        for b in range(batch_sz):
            for i in range(T_q):
                # Load Q_i from global memory
                q_start = i * B_Q
                q_end = q_start + B_Q
                Q_i = Q[b][q_start : q_end :]
                # Initialize O_i and L_i
                O_ij = O[b][q_start : q_end :]
                L_ij = L[b][q_start : q_end]
                m_ij = torch.full((B_Q,), float('-inf'), device=device, dtype=dtype)

                scale = 1 / math.sqrt(d)

                for j in range(T_k):
                    # Load K_j and V_j from global memory
                    k_start = j * B_K
                    k_end = k_start + B_K
                    K_j = K[b][k_start : k_end :]
                    V_j = V[b][k_start : k_end :]

                    S_ij = einsum(Q_i, K_j, "B_q d, B_k d -> B_q B_k") * scale
                    prev_mij = m_ij
                    m_ij = torch.max(m_ij, torch.max(S_ij, dim=-1).values)

                    # S_ij has shape B_q, B_k
                    # m_ij has shape B_q,
                    P_ij = torch.exp(S_ij - rearrange(m_ij, "(B_q x) -> B_q x", x=1))

                    L_ij = torch.exp(prev_mij - m_ij) * L_ij + torch.sum(P_ij, dim=-1)
                    O_ij = einsum(torch.diag(torch.exp(prev_mij - m_ij)), O_ij, "B_q B_q, B_q d -> B_q d") + einsum(P_ij, V_j, "B_q B_k, B_k d -> B_q d")
                
                # Compute Oi and Li
                O_ij = einsum(torch.diag(1 / L_ij), O_ij, "B_q B_q, B_q d -> B_q d")
                L_ij = m_ij + torch.log(L_ij)

                # Write Oi to global memory as the i-th tile of O
                O[b][q_start : q_end :] = O_ij
                # Write Li to global memory as the i-th tile of L
                L[b][q_start : q_end] = L_ij

        ctx.save_for_backward(L, Q, K, V, O)

        return O


    @staticmethod
    def backward():
        raise NotImplementedError


