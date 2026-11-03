# CUDA-graph capture/replay of the decode indexer step on Thor:
# paged MQA metadata + paged MQA logits + top-k, compared against eager on the same buffers,
# then again after rewriting the captured input buffers with a different batch.
# Tolerance: /data/vglm/m1/indexer-tolerances.md row 15.
import pytest
import torch

import vllm._custom_ops  # noqa: F401
from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits, get_num_sms, get_paged_mqa_logits_metadata

from test_deepgemm_attention_thor import D, MAX_MODEL_LEN, PAGE, make_paged_cache

K = 2048
WS = 1024 * 1024


def batch_contents(lens, next_n, heads, page_map, seed, num_pages, max_pages):
    g = torch.Generator(device="cuda").manual_seed(seed)
    k_list = [torch.randn(n, D, device="cuda", dtype=torch.bfloat16, generator=g) for n in lens]
    cache, bt = make_paged_cache(k_list, page_map, extra_pages=num_pages - sum((n + PAGE - 1) // PAGE for n in lens), seed=seed)
    bt_full = torch.zeros(len(lens), max_pages, dtype=torch.int32, device="cuda")
    bt_full[:, : bt.shape[1]] = bt
    B = len(lens)
    q = torch.randn(B, next_n, heads, D, device="cuda", dtype=torch.bfloat16, generator=g).to(torch.float8_e4m3fn)
    w = torch.randn(B * next_n, heads, device="cuda", dtype=torch.float32, generator=g)
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    ctx2d = (seq[:, None] - next_n + 1 + torch.arange(next_n, device="cuda", dtype=torch.int32)).contiguous()
    return dict(cache=cache, bt=bt_full, q=q, w=w, ctx2d=ctx2d)


def make_step(buf, selector):
    ws = torch.empty(WS, dtype=torch.uint8, device="cuda")
    rows = buf["q"].shape[0] * buf["q"].shape[1]
    out = torch.full((rows, K), -7, dtype=torch.int32, device="cuda")

    def step():
        meta = get_paged_mqa_logits_metadata(buf["ctx2d"], PAGE, get_num_sms())
        logits = fp8_fp4_paged_mqa_logits(
            (buf["q"], None), buf["cache"], buf["w"], buf["ctx2d"], buf["bt"], meta, MAX_MODEL_LEN, clean_logits=False)
        if selector == "cooperative":
            torch.ops._C.cooperative_topk(logits, buf["ctx2d"], out, ws, K, MAX_MODEL_LEN)
        else:
            torch.ops._C.persistent_topk(logits, buf["ctx2d"], out, ws, K, MAX_MODEL_LEN)
        return logits, out

    return step


def compare(label, g_logits, g_idx, e_logits, e_idx, ctx2d):
    lens = ctx2d.reshape(-1)
    valid = torch.arange(MAX_MODEL_LEN, device="cuda")[None] < lens[:, None]
    a = g_logits.masked_fill(~valid, 0).contiguous().view(torch.int32)
    b = e_logits.masked_fill(~valid, 0).contiguous().view(torch.int32)
    n_bad = int((a != b).sum())
    same_sets = torch.equal(g_idx.sort(1).values, e_idx.sort(1).values)
    same_order = torch.equal(g_idx, e_idx)
    print(f"GRAPH {label}: logits_bitwise_mismatches={n_bad} sets_equal={same_sets} order_equal={same_order}")
    assert n_bad == 0, f"{label}: {n_bad} logits differ"
    assert same_sets, f"{label}: top-k sets differ"


@pytest.mark.parametrize("heads", [32, 64], ids=["h32", "h64"])
@pytest.mark.parametrize("selector", ["cooperative", "persistent"])
@pytest.mark.parametrize("next_n", [1, 2], ids=["n1", "n2"])
@pytest.mark.parametrize("batch", [1, 4, 8], ids=["b1", "b4", "b8"])
def test_decode_indexer_graph_replay(batch, next_n, selector, heads):
    lens_a = ([16384, 2049, 4096, 1, 2047, 2048, 9000, 640] * 2)[:batch]
    lens_b = ([2050, 16384, 64, 3000, 16383, 2049, 1, 8191] * 2)[:batch]
    lens_a = [max(n, next_n) for n in lens_a]
    lens_b = [max(n, next_n) for n in lens_b]
    max_pages = MAX_MODEL_LEN // PAGE
    num_pages = sum((n + PAGE - 1) // PAGE for n in lens_a + lens_b) + 8
    buf = batch_contents(lens_a, next_n, heads, "shuffled", 1, num_pages, max_pages)
    step = make_step(buf, "cooperative" if selector == "cooperative" else "persistent")

    # Eager warm-up (JIT compile + module load must happen outside capture).
    e_logits, e_idx = step()
    e_logits, e_idx = e_logits.clone(), e_idx.clone()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(graph):
        g_logits, g_idx = step()
    graph.replay()
    torch.cuda.synchronize()
    compare(f"B={batch} n={next_n} {selector} h={heads} capture-contents", g_logits, g_idx, e_logits, e_idx, buf["ctx2d"])

    # Rewrite every captured input buffer with a different batch, replay, compare with eager.
    new = batch_contents(lens_b, next_n, heads, "interleaved", 2, num_pages, max_pages)
    for k_ in buf:
        buf[k_].copy_(new[k_])
    graph.replay()
    torch.cuda.synchronize()
    g2_logits, g2_idx = g_logits.clone(), g_idx.clone()
    e2_logits, e2_idx = step()
    torch.cuda.synchronize()
    compare(f"B={batch} n={next_n} {selector} h={heads} new-contents", g2_logits, g2_idx, e2_logits, e2_idx, buf["ctx2d"])
    # The new batch must actually change the result.
    assert not torch.equal(g2_idx.sort(1).values, e_idx.sort(1).values)
