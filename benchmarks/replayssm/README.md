# ReplaySSM benchmarks

## Paired conv1d and ReplaySSM PDL

`compare_conv1d_pdl.py` measures vLLM's causal-conv1d producer immediately
followed by FlashInfer ReplaySSM. It checks every output, state, cache, and ring
tracker bitwise against the fully serialized chain before reporting timings.

Run each launch policy in a clean process:

```bash
for mode in off producer-only consumer-only paired; do
  python benchmarks/replayssm/compare_conv1d_pdl.py \
    --pdl-mode "$mode" --output "pair_${mode}.json"
done
```

The default matrix covers batch sizes 1, 8, 16, 32, 64, and 128;
verification lengths 2, 4, and 8; and both ReplaySSM verify and flush paths.
FlashInfer native tactic tuning is enabled by default.

## Actual-model PDL throughput

`e2e_spec_decode_throughput.py` accepts a benchmark-only PDL override for the
conv1d to ReplaySSM chain. To hold the number of speculative steps constant,
use the synthetic sampler with an acceptance length of `num_spec + 1`:

```bash
python benchmarks/replayssm/e2e_spec_decode_throughput.py \
  --model-id /path/to/model --batch-size 16 --num-spec 3 \
  --synthetic-acceptance-length 4 --modes cache \
  --replayssm-backend flashinfer --replayssm-pdl off \
  --max-tokens 256 --warmup-s 10 --measure-repeats 3

python benchmarks/replayssm/e2e_spec_decode_throughput.py \
  --model-id /path/to/model --batch-size 16 --num-spec 3 \
  --synthetic-acceptance-length 4 --modes cache \
  --replayssm-backend flashinfer --replayssm-pdl on \
  --max-tokens 256 --warmup-s 10 --measure-repeats 3
```

Synthetic acceptance is a performance control, not an accuracy check. Omit it
and run `--modes spec,cache` for token parity with native MTP.
