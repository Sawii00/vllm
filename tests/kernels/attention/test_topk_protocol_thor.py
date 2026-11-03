# Top-k selection protocol for the DSA indexer on Thor (plan correctness protocol).
# Tolerances: /data/vglm/m1/indexer-tolerances.md rows 11-14.
import pytest
import torch

import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    get_num_sms,
    get_paged_mqa_logits_metadata,
)

from test_deepgemm_attention_thor import MAX_MODEL_LEN, D, PAGE, make_paged_cache, quant_k_rows

K = 2048
WS = 1024 * 1024
DECODE_SELECTORS = ["per_row", "persistent", "cooperative"]


# ---------------------------------------------------------------- selectors

def run_decode_selector(name, logits, seq_lens, next_n):
    """seq_lens: (B,) request lengths. Row b*next_n+t has length seq_lens[b]-next_n+1+t."""
    rows = logits.shape[0]
    out = torch.full((rows, K), -7, dtype=torch.int32, device="cuda")
    row_lens = (seq_lens[:, None] - next_n + 1 + torch.arange(next_n, device="cuda", dtype=torch.int32)).reshape(-1)
    max_len = int(seq_lens.max())
    if name == "per_row":
        torch.ops._C.top_k_per_row_decode(logits, next_n, seq_lens, out, rows, logits.stride(0), logits.stride(1), K)
    elif name == "persistent":
        ws = torch.empty(WS, dtype=torch.uint8, device="cuda")
        torch.ops._C.persistent_topk(logits, row_lens.contiguous(), out, ws, K, max_len)
    elif name == "cooperative":
        assert rows <= 64 and logits.stride(0) % 4 == 0
        ws = torch.empty(WS, dtype=torch.uint8, device="cuda")
        torch.ops._C.cooperative_topk(logits, row_lens.contiguous(), out, ws, K, max_len)
    else:
        raise ValueError(name)
    torch.cuda.synchronize()
    return out, torch.zeros_like(row_lens), row_lens


def run_prefill_selector(logits, ks, ke):
    rows = logits.shape[0]
    out = torch.full((rows, K), -7, dtype=torch.int32, device="cuda")
    torch.ops._C.top_k_per_row_prefill(logits, ks, ke, out, rows, logits.stride(0), logits.stride(1), K)
    torch.cuda.synchronize()
    return out


# ---------------------------------------------------------------- checks

def window_masked(scores, starts, ends):
    cols = torch.arange(scores.shape[1], device="cuda")
    inside = (cols[None] >= starts[:, None]) & (cols[None] < ends[:, None])
    return scores.masked_fill(~inside, float("-inf")), inside


def ref_topk_row_local(scores, starts, ends):
    """Independent reference: torch.topk inside each row window, row-local indices, -1 padding."""
    masked, _ = window_masked(scores, starts, ends)
    vals, idx = masked.topk(K, dim=-1)
    lens = (ends - starts).clamp(min=0)
    keep = torch.arange(K, device="cuda")[None] < lens[:, None]
    return torch.where(keep, idx - starts[:, None], -1).to(torch.int32), vals


def check_structure(out, starts, ends, label):
    """Valid range, uniqueness, expected valid count and -1 padding, per row."""
    lens = (ends - starts).clamp(min=0)
    cnt = lens.clamp(max=K)
    valid = out >= 0
    assert ((out == -1) | valid).all(), f"{label}: indices other than -1 below 0"
    assert torch.equal(valid.sum(1), cnt), f"{label}: valid counts differ"
    assert (out[valid] < lens[:, None].expand_as(out)[valid]).all(), f"{label}: index >= row length"
    srt = torch.where(valid, out, torch.iinfo(torch.int32).max).sort(1).values
    dup = (srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] != torch.iinfo(torch.int32).max)
    assert not dup.any(), f"{label}: duplicate valid indices in rows {dup.any(1).nonzero().flatten().tolist()[:10]}"


def sorted_sets(idx):
    return torch.where(idx >= 0, idx, -1).sort(1).values


def assert_exact_sets(out, ref, label, scores=None, starts=None):
    """Exact selected-set equality; if `scores` is given, a swap is allowed only between
    bit-identical score values (true ties)."""
    a, b = sorted_sets(out), sorted_sets(ref)
    bad = (a != b).any(1).nonzero().flatten().tolist()
    if bad and scores is not None:
        still_bad = []
        for r in bad:
            sa = set(out[r][out[r] >= 0].tolist())
            sb = set(ref[r][ref[r] >= 0].tolist())
            row = scores[r, int(starts[r]):]
            va = sorted(row[list(sa - sb)].tolist())
            vb = sorted(row[list(sb - sa)].tolist())
            if va != vb:
                still_bad.append(r)
        bad = still_bad
    assert not bad, f"{label}: {len(bad)} rows with different selected sets, first rows {bad[:10]}"


def separated_scores(rows, width, starts, ends, seed, garbage):
    """Distinct values with an absolute gap of 0.01 inside each row window; `garbage` outside."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    perm = torch.rand(rows, width, device="cuda", generator=g).argsort(1).float()
    vals = (perm - width / 2) * 0.01
    _, inside = window_masked(vals, starts, ends)
    return vals.masked_fill(~inside, garbage)


# ---------------------------------------------------------------- (a) synthetic, separated

DECODE_LENS = {
    "n2049": [2049], "n4096": [4096], "n16384": [16384], "short": [1, 2047, 2048],
    "mixed": [16384, 1, 2049, 4096, 2047, 2048, 9000, 65],
}


@pytest.mark.parametrize("garbage", [1e30, float("inf")], ids=["big", "inf"])
@pytest.mark.parametrize("next_n", [1, 2], ids=["n1", "n2"])
@pytest.mark.parametrize("batch", [1, 4, 8], ids=["b1", "b4", "b8"])
@pytest.mark.parametrize("lens_id", list(DECODE_LENS))
@pytest.mark.parametrize("selector", DECODE_SELECTORS)
def test_decode_selector_separated(selector, lens_id, batch, next_n, garbage):
    base = DECODE_LENS[lens_id]
    seq = [max(base[i % len(base)], next_n) for i in range(batch)]
    seq_lens = torch.tensor(seq, dtype=torch.int32, device="cuda")
    rows = batch * next_n
    row_lens = (seq_lens[:, None] - next_n + 1 + torch.arange(next_n, device="cuda", dtype=torch.int32)).reshape(-1)
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    scores = separated_scores(rows, MAX_MODEL_LEN, starts, row_lens, seed=len(seq) * 31 + next_n, garbage=garbage)
    out, s, e = run_decode_selector(selector, scores, seq_lens, next_n)
    label = f"{selector} {lens_id} B={batch} next_n={next_n}"
    check_structure(out, s, e, label)
    ref, _ = ref_topk_row_local(scores, s, e)
    assert_exact_sets(out, ref, label)

    # Different request ordering: results must follow their rows.
    perm = torch.randperm(batch, generator=torch.Generator().manual_seed(3)).cuda()
    row_perm = (perm[:, None] * next_n + torch.arange(next_n, device="cuda")).reshape(-1)
    out_p, _, _ = run_decode_selector(selector, scores[row_perm].contiguous(), seq_lens[perm].contiguous(), next_n)
    assert_exact_sets(out_p, out[row_perm], label + " permuted")


PREFILL_LAYOUTS = {
    # (request lengths packed in one K workspace, chunk offset: query rows are the last `m` of each)
    "single2049": ([2049], None), "single4096": ([4096], None), "single16384": ([16384], 1024),
    "packed": ([2049, 1, 4096, 2047, 2048], None),
}


def prefill_rows(lens, tail):
    ks, ke, off = [], [], 0
    for n in lens:
        first = 0 if tail is None else max(0, n - tail)
        ks += [off] * (n - first)
        ke += list(range(off + first + 1, off + n + 1))
        off += n
    return torch.tensor(ks, dtype=torch.int32, device="cuda"), torch.tensor(ke, dtype=torch.int32, device="cuda"), off


@pytest.mark.parametrize("shuffle_rows", [False, True], ids=["ordered", "shuffled"])
@pytest.mark.parametrize("layout", list(PREFILL_LAYOUTS))
def test_prefill_selector_separated(layout, shuffle_rows):
    lens, tail = PREFILL_LAYOUTS[layout]
    ks, ke, width = prefill_rows(lens, tail)
    if shuffle_rows:
        p = torch.randperm(ks.numel(), generator=torch.Generator().manual_seed(11)).cuda()
        ks, ke = ks[p].contiguous(), ke[p].contiguous()
    scores = separated_scores(ks.numel(), width, ks, ke, seed=17, garbage=1e30)
    out = run_prefill_selector(scores, ks, ke)
    check_structure(out, ks, ke, f"prefill {layout}")
    ref, _ = ref_topk_row_local(scores, ks, ke)
    assert_exact_sets(out, ref, f"prefill {layout}")


# ---------------------------------------------------------------- (b)-(d) real kernel scores

def band_report(label, kern, sim, true_scores, starts, ends, out):
    """Score check, then membership within the predeclared band around the kth score."""
    masked_sim, inside = window_masked(sim, starts, ends)
    eps = 1e-4 * sim.abs().masked_fill(~inside, 0).amax(1)
    err = (kern - sim).abs().masked_fill(~inside, 0).amax(1)
    over = (err > eps).nonzero().flatten().tolist()
    lens = ends - starts
    ref, ref_vals = ref_topk_row_local(sim, starts, ends)
    true_ref, _ = ref_topk_row_local(true_scores, starts, ends)
    long_rows = (lens > K).nonzero().flatten().tolist()
    in_band = out_band = 0
    margins, overlaps, worst = [], [], []
    for r in long_rows:
        s = int(starts[r])
        row = masked_sim[r]
        top = row.topk(K + 1).values
        kth = top[K - 1]
        margins.append(float((top[K - 1] - top[K]) / eps[r]))
        got = set(out[r][out[r] >= 0].tolist())
        exp = set(ref[r][ref[r] >= 0].tolist())
        diff = list(got ^ exp)
        if diff:
            d = (row[torch.tensor(diff, device="cuda") + s] - kth).abs()
            inb = d <= 2 * eps[r]
            in_band += int(inb.sum())
            out_band += int((~inb).sum())
            if (~inb).any():
                worst.append((r, float(d.max() / eps[r]), len(diff)))
        tset = set(true_ref[r][true_ref[r] >= 0].tolist())
        overlaps.append(len(got & tset) / K)
    rel_err = (err / eps.clamp(min=1e-30)).max().item() * 1e-4
    msg = (f"BAND {label}: rows={kern.shape[0]} long_rows={len(long_rows)} max_rel_score_err={rel_err:.2e} "
           f"rows_over_eps={len(over)} swaps_in_band={in_band} swaps_out_of_band={out_band} "
           f"min_margin_eps={min(margins) if margins else float('nan'):.3g} "
           f"median_margin_eps={sorted(margins)[len(margins) // 2] if margins else float('nan'):.3g} "
           f"overlap_vs_true_min={min(overlaps) if overlaps else float('nan'):.4f} "
           f"overlap_vs_true_mean={sum(overlaps) / max(1, len(overlaps)):.4f}")
    print(msg)
    for w in worst:
        print(f"  OUT-OF-BAND row={w[0]} max|s-kth|/eps={w[1]:.3g} n_diff={w[2]}")
    assert not over, f"{label}: score error above eps in rows {over[:10]}"
    assert out_band == 0, f"{label}: {out_band} membership changes outside the band"


def paged_inputs(seq, next_n, heads, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    k_list = [torch.randn(n, D, device="cuda", dtype=torch.bfloat16, generator=g) for n in seq]
    cache, bt = make_paged_cache(k_list, "shuffled")
    B = len(seq)
    q = torch.randn(B, next_n, heads, D, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(B * next_n, heads, device="cuda", dtype=torch.float32, generator=g)
    return k_list, cache, bt, q, w


def paged_scores(seq, next_n, heads, seed):
    k_list, cache, bt, q, w = paged_inputs(seq, next_n, heads, seed)
    q_fp8 = q.to(torch.float8_e4m3fn)
    seq_lens = torch.tensor(seq, dtype=torch.int32, device="cuda")
    ctx2d = (seq_lens[:, None] - next_n + 1 + torch.arange(next_n, device="cuda", dtype=torch.int32)).contiguous()
    meta = get_paged_mqa_logits_metadata(ctx2d, PAGE, get_num_sms())
    kern = fp8_fp4_paged_mqa_logits((q_fp8, None), cache, w, ctx2d, bt, meta, MAX_MODEL_LEN, clean_logits=False)

    def ref(qq, ks_):
        out = torch.zeros(len(seq) * next_n, MAX_MODEL_LEN, device="cuda")
        for b, k in enumerate(ks_):
            s = torch.einsum("thd,nd->thn", qq[b].float(), k.float()).relu()
            out[b * next_n:(b + 1) * next_n, : k.shape[0]] = (s * w[b * next_n:(b + 1) * next_n, :, None]).sum(1)
        return out

    sim = ref(q_fp8.to(torch.bfloat16), [quant_k_rows(k)[2] for k in k_list])
    true_s = ref(q, k_list)
    return kern, sim, true_s, seq_lens


REAL_DECODE = {"n2049": [2049] * 4, "n4096": [4096] * 4, "n16384": [16384] * 4,
               "mixed": [16384, 2049, 4096, 2047, 1, 9000, 2048, 3000]}


@pytest.mark.parametrize("heads", [32, 64], ids=["h32", "h64"])
@pytest.mark.parametrize("next_n", [1, 2], ids=["n1", "n2"])
@pytest.mark.parametrize("case", list(REAL_DECODE))
@pytest.mark.parametrize("selector", DECODE_SELECTORS)
def test_decode_real_scores(selector, case, next_n, heads):
    seq = REAL_DECODE[case]
    kern, sim, true_s, seq_lens = paged_scores(seq, next_n, heads, seed=list(REAL_DECODE).index(case) * 101 + next_n)
    out, s, e = run_decode_selector(selector, kern, seq_lens, next_n)
    label = f"decode {selector} {case} next_n={next_n} h={heads}"
    check_structure(out, s, e, label)
    # (b) selector on the identical kernel scores
    ref_k, _ = ref_topk_row_local(kern, s, e)
    assert_exact_sets(out, ref_k, label + " vs torch.topk(kernel scores)", scores=kern, starts=s)
    # (c) + (d)
    band_report(label, kern, sim, true_s, s, e, out)


@pytest.mark.parametrize("heads", [32, 64], ids=["h32", "h64"])
@pytest.mark.parametrize("n", [2049, 4096, 16384])
def test_prefill_real_scores(n, heads):
    # Chunked prefill: the last 512 query rows of an n-token prompt.
    m = 512
    g = torch.Generator(device="cuda").manual_seed(n)
    q = torch.randn(m, heads, D, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(n, D, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(m, heads, device="cuda", dtype=torch.float32, generator=g)
    ks = torch.zeros(m, dtype=torch.int32, device="cuda")
    ke = torch.arange(n - m + 1, n + 1, dtype=torch.int32, device="cuda")
    q_fp8 = q.to(torch.float8_e4m3fn)
    k_fp8, k_sf, k_sim = quant_k_rows(k)
    kern = fp8_fp4_mqa_logits((q_fp8, None), (k_fp8, k_sf), w, ks, ke, clean_logits=False)

    def ref(qq, kk):
        s = torch.einsum("mhd,nd->mhn", qq.float(), kk.float()).relu()
        return (s * w[:, :, None]).sum(1)

    sim, true_s = ref(q_fp8.to(torch.bfloat16), k_sim), ref(q, k)
    out = run_prefill_selector(kern, ks, ke)
    label = f"prefill n={n} h={heads}"
    check_structure(out, ks, ke, label)
    ref_k, _ = ref_topk_row_local(kern, ks, ke)
    assert_exact_sets(out, ref_k, label + " vs torch.topk(kernel scores)", scores=kern, starts=ks)
    band_report(label, kern, sim, true_s, ks, ke, out)
