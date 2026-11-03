# Thor (sm_110) extension of test_deepgemm_attention.py at GLM-5.3 / DeepSeek-V3.2 indexer dims.
# Tolerances: /data/vglm/m1/indexer-tolerances.md rows 1-5.
import pytest
import torch

from vllm.utils.deep_gemm import (
    _ceil_to_ue8m0,
    calc_diff,
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    get_num_sms,
    get_paged_mqa_logits_metadata,
)

D = 128
PAGE = 64
MAX_MODEL_LEN = 16384
CONTEXTS = [1, 2047, 2048, 2049, 4096, 16384]


def quant_k_rows(k: torch.Tensor):
    """Per-token fp8 e4m3 with a UE8M0 (power-of-two) fp32 scale, as in the vLLM indexer K cache.

    Returns (fp8 values, fp32 scale per row, fp32 dequantized values)."""
    sf = _ceil_to_ue8m0(k.abs().float().amax(-1, keepdim=True).clamp(1e-4) / 448.0)
    k_fp8 = (k.float() / sf).to(torch.float8_e4m3fn)
    return k_fp8, sf.squeeze(-1).contiguous(), k_fp8.float() * sf


def make_paged_cache(k_tokens_per_req, page_map: str, extra_pages: int = 3, seed: int = 7):
    """Write each request's K rows into 64-token pages laid out as
    [64 x 128 fp8][64 x fp32 scale] = 8448 B per page (view [pages, 64, 1, 132])."""
    pages_per_req = [(k.shape[0] + PAGE - 1) // PAGE for k in k_tokens_per_req]
    total = sum(pages_per_req) + extra_pages
    if page_map == "shuffled":
        order = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).tolist()
    elif page_map == "reversed":
        order = list(range(total))[::-1]
    elif page_map == "interleaved":
        # Page j of every request before page j+1 of any request.
        slots = [[None] * p for p in pages_per_req]
        nxt = 0
        for j in range(max(pages_per_req)):
            for i, p in enumerate(pages_per_req):
                if j < p:
                    slots[i][j] = nxt
                    nxt += 1
        order = [s for req in slots for s in req] + list(range(nxt, total))
    else:
        raise ValueError(page_map)
    # Finite garbage everywhere first (fp8 bytes < 0x7E, large scales): unused pages and
    # the tails of partially filled pages must not leak into valid logits.
    cache = torch.randint(0, 0x7E, (total, PAGE * (D + 4)), dtype=torch.uint8, device="cuda")
    cache[:, PAGE * D:] = torch.full((total, PAGE), 1e4, device="cuda").view(torch.uint8)
    bt = torch.zeros(len(k_tokens_per_req), max(pages_per_req), dtype=torch.int32, device="cuda")
    pos = 0
    for i, k in enumerate(k_tokens_per_req):
        k_fp8, sf, _ = quant_k_rows(k)
        for j in range(pages_per_req[i]):
            phys = order[pos]
            pos += 1
            bt[i, j] = phys
            rows = k_fp8[j * PAGE:(j + 1) * PAGE]
            n = rows.shape[0]
            cache[phys, : n * D] = rows.contiguous().view(torch.uint8).view(-1)
            cache[phys, PAGE * D: PAGE * D + 4 * n] = sf[j * PAGE: j * PAGE + n].contiguous().view(torch.uint8)
    return cache.view(total, PAGE, 1, D + 4), bt


def paged_ref(q, k_list, weights, ctx2d, width=MAX_MODEL_LEN):
    """fp32 reference: logits[b*next_n + t, s] = sum_h w[h] * relu(q[h] . k[s]) for s < ctx2d[b, t]."""
    B, next_n, H, _ = q.shape
    out = torch.zeros(B * next_n, width, device="cuda")
    valid = torch.zeros_like(out, dtype=torch.bool)
    for b in range(B):
        s = torch.einsum("thd,nd->thn", q[b].float(), k_list[b].float()).relu()
        lg = (s * weights[b * next_n:(b + 1) * next_n, :, None]).sum(1)
        for t in range(next_n):
            n = int(ctx2d[b, t])
            out[b * next_n + t, :n] = lg[t, :n]
            valid[b * next_n + t, :n] = True
    return out, valid


def mixed_lens(batch: int):
    choices = CONTEXTS + [65, 127, 640, 9000]
    gen = torch.Generator().manual_seed(batch)
    lens = [choices[i] for i in torch.randint(0, len(choices), (batch,), generator=gen).tolist()]
    if batch > 1:
        lens[0], lens[-1] = 16384, 1
    return lens


PAGED_CASES = [("uniform", c) for c in CONTEXTS] + [("mixed", 0)]


@pytest.mark.parametrize("heads", [32, 64], ids=["h32", "h64"])
@pytest.mark.parametrize("next_n", [1, 2], ids=["n1", "n2"])
@pytest.mark.parametrize("page_map", ["shuffled", "reversed", "interleaved"])
@pytest.mark.parametrize("batch", [1, 4, 8], ids=["b1", "b4", "b8"])
@pytest.mark.parametrize("mode,ctx", PAGED_CASES, ids=[f"{m}{c}" for m, c in PAGED_CASES])
def test_paged_mqa_logits_thor(mode, ctx, batch, page_map, next_n, heads):
    g = torch.Generator(device="cuda").manual_seed(1000 + batch * 10 + next_n)
    lens = [ctx] * batch if mode == "uniform" else mixed_lens(batch)
    lens = [max(n, next_n) for n in lens]
    k_list = [torch.randn(n, D, device="cuda", dtype=torch.bfloat16, generator=g) for n in lens]
    cache, bt = make_paged_cache(k_list, page_map)
    q = torch.randn(batch, next_n, heads, D, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(batch * next_n, heads, device="cuda", dtype=torch.float32, generator=g)
    q_fp8 = q.to(torch.float8_e4m3fn)
    ctx_t = torch.tensor(lens, dtype=torch.int32, device="cuda")
    ctx2d = (ctx_t[:, None] - next_n + 1 + torch.arange(next_n, device="cuda", dtype=torch.int32)).contiguous()
    meta = get_paged_mqa_logits_metadata(ctx2d, PAGE, get_num_sms())
    logits = fp8_fp4_paged_mqa_logits((q_fp8, None), cache, w, ctx2d, bt, meta, MAX_MODEL_LEN, clean_logits=False)
    assert logits.shape == (batch * next_n, MAX_MODEL_LEN) and logits.dtype == torch.float32

    ref, valid = paged_ref(q, k_list, w, ctx2d)
    sim, _ = paged_ref(q_fp8.to(torch.bfloat16), [quant_k_rows(k)[2] for k in k_list], w, ctx2d)
    got = logits.masked_fill(~valid, 0)
    assert torch.isfinite(got).all(), "non-finite logits inside the valid range"
    d_true, d_sim = calc_diff(got, ref), calc_diff(got, sim)
    # Diagnostic only: how far the test's own FP8 quantization moves the inputs.
    d_quant = calc_diff(sim, ref)
    print(f"PAGED h={heads} n={next_n} map={page_map} B={batch} lens={lens} "
          f"diff_true={d_true:.3e} diff_sim={d_sim:.3e} diff_sim_vs_true={d_quant:.3e}")
    assert d_true < 1e-3, f"{d_true=}"
    assert d_sim < 5e-6, f"{d_sim=}"


def dense_case(kind):
    """(q rows, k rows, ks, ke) for packed prefill as vLLM builds it (ks/ke index the K workspace)."""
    if kind.startswith("single"):
        n = int(kind[6:])
        return n, n, torch.zeros(n, dtype=torch.int32), torch.arange(1, n + 1, dtype=torch.int32)
    if kind == "packed":
        ks, ke, off = [], [], 0
        for n in [2049, 1, 4096, 2047]:
            ks += [off] * n
            ke += list(range(off + 1, off + n + 1))
            off += n
        return off, off, torch.tensor(ks, dtype=torch.int32), torch.tensor(ke, dtype=torch.int32)
    if kind == "chunked":
        # Last 2048-token chunk of a 16384-token prompt; earlier K already in the workspace.
        total, m = 16384, 2048
        ke = torch.arange(total - m + 1, total + 1, dtype=torch.int32)
        return m, total, torch.zeros(m, dtype=torch.int32), ke
    raise ValueError(kind)


DENSE_KINDS = [f"single{c}" for c in CONTEXTS] + ["packed", "chunked"]


@pytest.mark.parametrize("heads", [32, 64], ids=["h32", "h64"])
@pytest.mark.parametrize("clean_logits", [False, True], ids=["noclean", "clean"])
@pytest.mark.parametrize("kind", DENSE_KINDS)
def test_mqa_logits_thor(kind, clean_logits, heads):
    g = torch.Generator(device="cuda").manual_seed(5)
    m, n, ks, ke = dense_case(kind)
    ks, ke = ks.cuda(), ke.cuda()
    q = torch.randn(m, heads, D, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(n, D, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(m, heads, device="cuda", dtype=torch.float32, generator=g)
    q_fp8 = q.to(torch.float8_e4m3fn)
    k_fp8, k_sf, k_sim = quant_k_rows(k)
    logits = fp8_fp4_mqa_logits((q_fp8, None), (k_fp8, k_sf), w, ks, ke, clean_logits=clean_logits)
    assert logits.shape == (m, n)
    cols = torch.arange(n, device="cuda")
    valid = (cols[None] >= ks[:, None]) & (cols[None] < ke[:, None])

    def ref(qq, kk):
        out = torch.empty(m, n, device="cuda")
        step = max(1, (1 << 28) // (n * heads * 4))
        for r0 in range(0, m, step):
            s = torch.einsum("mhd,nd->mhn", qq[r0:r0 + step].float(), kk.float()).relu()
            out[r0:r0 + step] = (s * w[r0:r0 + step, :, None]).sum(1)
        return out.masked_fill(~valid, 0)

    if clean_logits:
        assert torch.equal(logits == float("-inf"), ~valid), "-inf mask mismatch"
    got = logits.masked_fill(~valid, 0)
    assert torch.isfinite(got).all()
    r_true, r_sim = ref(q, k), ref(q_fp8.to(torch.bfloat16), k_sim)
    d_true, d_sim = calc_diff(got, r_true), calc_diff(got, r_sim)
    print(f"DENSE h={heads} kind={kind} clean={clean_logits} diff_true={d_true:.3e} diff_sim={d_sim:.3e} "
          f"diff_sim_vs_true={calc_diff(r_sim, r_true):.3e}")
    assert d_true < 1e-3, f"{d_true=}"
    assert d_sim < 5e-6, f"{d_sim=}"
