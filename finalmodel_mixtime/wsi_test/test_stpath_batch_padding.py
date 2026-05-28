"""
Test that the batch-padding fix in extract_embeddings.py handles
last-chunk sizes that are smaller than stpath_batch_size.

Runs entirely on CPU with no real STPath models.
"""

import sys
import types
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Build a minimal fake STPath stack so we can import the patched loop logic
# without needing the real conda env.
# ---------------------------------------------------------------------------

REQUIRED_BS = 256  # STPath's internal hard requirement


def fake_prediction_head(img_tokens, coords, ge_tokens, batch_idx,
                         tech_tokens, organ_tokens, return_all=False):
    """Mimics STPath's model – raises if batch size != REQUIRED_BS."""
    n = img_tokens.shape[0]
    if n != REQUIRED_BS:
        raise RuntimeError(
            f"The size of tensor a ({n}) must match the size of tensor b "
            f"({REQUIRED_BS}) at non-singleton dimension 0 "
            f"please correct them by using an adaptive batch size, "
            f"starting from around {REQUIRED_BS}/"
        )
    # Return fake gene-expression output [N, 38984] + dummy aux
    return torch.zeros(n, 38984), None


class FakeModel:
    def prediction_head(self, **kwargs):
        return fake_prediction_head(**kwargs)


class FakeAgent:
    model = FakeModel()

    def _generate_masked_ge_tokens(self, n):
        return torch.zeros(n, 10)

    def _generate_pad_tech_tokens(self, n):
        return torch.zeros(n, 4, dtype=torch.long)


# ---------------------------------------------------------------------------
# The patched loop (copy of the fixed code from extract_embeddings.py)
# ---------------------------------------------------------------------------

def run_stpath_loop(N_patches, bs, feats_gp_np, coords_norm, organ_id,
                    agent, device="cpu"):
    """Run the batch-padded STPath inference loop."""
    all_preds = []
    for start in range(0, N_patches, bs):
        end    = min(start + bs, N_patches)
        actual = end - start

        c_chunk   = coords_norm[start:end]
        f_chunk   = torch.from_numpy(feats_gp_np[start:end])
        masked_ge = agent._generate_masked_ge_tokens(actual)
        tech_ids  = agent._generate_pad_tech_tokens(actual)
        organ_ids = torch.full((actual,), organ_id, dtype=torch.long)
        batch_idx = torch.zeros(actual, dtype=torch.long)

        if actual < bs:
            pad       = bs - actual
            c_chunk   = torch.cat([c_chunk,   c_chunk[:1].expand(pad, -1)], dim=0)
            f_chunk   = torch.cat([f_chunk,   f_chunk[:1].expand(pad, -1)], dim=0)
            masked_ge = torch.cat([masked_ge, masked_ge[:1].expand(pad, -1)], dim=0)
            tech_ids  = torch.cat([tech_ids,  tech_ids[:1].expand(pad, -1)], dim=0)
            organ_ids = torch.cat([organ_ids, organ_ids[:1].expand(pad)], dim=0)
            batch_idx = torch.cat([batch_idx, batch_idx[:1].expand(pad)], dim=0)

        with torch.no_grad():
            pred_chunk, _ = agent.model.prediction_head(
                img_tokens=f_chunk,
                coords=c_chunk,
                ge_tokens=masked_ge,
                batch_idx=batch_idx,
                tech_tokens=tech_ids,
                organ_tokens=organ_ids,
                return_all=True,
            )
        all_preds.append(pred_chunk[:actual])

    return torch.cat(all_preds, dim=0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_exact_multiple():
    """N = 512 = 2 * bs → no padding needed."""
    N, bs = 512, 256
    feats  = np.zeros((N, 1536), dtype=np.float32)
    coords = torch.zeros(N, 2)
    agent  = FakeAgent()

    out = run_stpath_loop(N, bs, feats, coords, organ_id=0, agent=agent)
    assert out.shape == (N, 38984), f"Expected ({N}, 38984), got {out.shape}"
    print(f"  PASS  test_exact_multiple  output={tuple(out.shape)}")


def test_partial_last_batch():
    """N = 87 < bs = 256 → single last-chunk with padding."""
    N, bs = 87, 256
    feats  = np.zeros((N, 1536), dtype=np.float32)
    coords = torch.zeros(N, 2)
    agent  = FakeAgent()

    out = run_stpath_loop(N, bs, feats, coords, organ_id=0, agent=agent)
    assert out.shape == (N, 38984), f"Expected ({N}, 38984), got {out.shape}"
    print(f"  PASS  test_partial_last_batch  output={tuple(out.shape)}")


def test_multiple_plus_remainder():
    """N = 265 = 256 + 9 → one full batch + one padded last batch."""
    N, bs = 265, 256
    feats  = np.zeros((N, 1536), dtype=np.float32)
    coords = torch.zeros(N, 2)
    agent  = FakeAgent()

    out = run_stpath_loop(N, bs, feats, coords, organ_id=0, agent=agent)
    assert out.shape == (N, 38984), f"Expected ({N}, 38984), got {out.shape}"
    print(f"  PASS  test_multiple_plus_remainder  output={tuple(out.shape)}")


def test_single_patch():
    """N = 1 → extreme edge case."""
    N, bs = 1, 256
    feats  = np.zeros((N, 1536), dtype=np.float32)
    coords = torch.zeros(N, 2)
    agent  = FakeAgent()

    out = run_stpath_loop(N, bs, feats, coords, organ_id=0, agent=agent)
    assert out.shape == (N, 38984), f"Expected ({N}, 38984), got {out.shape}"
    print(f"  PASS  test_single_patch  output={tuple(out.shape)}")


def test_original_bug_without_fix():
    """Confirm the original code (no padding) raises with N=87, bs=256."""
    N, bs = 87, 256
    feats  = np.zeros((N, 1536), dtype=np.float32)
    coords = torch.zeros(N, 2)
    agent  = FakeAgent()

    raised = False
    try:
        # ---- original loop, no padding ----
        for start in range(0, N, bs):
            end    = min(start + bs, N)
            actual = end - start
            f_chunk   = torch.from_numpy(feats[start:end])
            c_chunk   = coords[start:end]
            masked_ge = agent._generate_masked_ge_tokens(actual)
            tech_ids  = agent._generate_pad_tech_tokens(actual)
            organ_ids = torch.full((actual,), 0, dtype=torch.long)
            batch_idx = torch.zeros(actual, dtype=torch.long)
            agent.model.prediction_head(
                img_tokens=f_chunk, coords=c_chunk,
                ge_tokens=masked_ge, batch_idx=batch_idx,
                tech_tokens=tech_ids, organ_tokens=organ_ids,
                return_all=True,
            )
    except RuntimeError as e:
        raised = True
        print(f"  PASS  test_original_bug_without_fix  got expected error: {e}")

    if not raised:
        raise AssertionError("Expected RuntimeError was NOT raised — something changed.")


if __name__ == "__main__":
    print("Running STPath batch-padding tests ...\n")
    test_original_bug_without_fix()
    test_partial_last_batch()
    test_exact_multiple()
    test_multiple_plus_remainder()
    test_single_patch()
    print("\nAll tests passed.")
