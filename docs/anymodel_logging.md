# AnyModel Logging — Execution Sequence

All log statements are at `DEBUG` level via `logger.debug`.

**Enable with:**
```bash
VLLM_LOGGING_LEVEL=DEBUG
```
or in Python:
```python
import logging
logging.getLogger("vllm.model_executor.models.anymodel").setLevel(logging.DEBUG)
```

## Call Sequence

### Model Initialization

```
→ AnyModel.__new__
  → _resolve_arch
    → _arch_info_from_config       (tries hf_config.anymodel_arch_info)
    ← _arch_info_from_config
    → _validate_config_arch_info   (if anymodel_arch_info was present)
    ← _validate_config_arch_info
    ← _resolve_arch
  → _get_or_create_wrapper         (logs cache hit/miss)
    → _make_wrapper_cls            (on cache miss only)
    ← _make_wrapper_cls
  ← AnyModel.__new__

→ AnyModel.__init__
  (calls super().__init__ — base model builds all layers with global config)
  → _patch_anymodel_layers
      layer N: has_overrides=True/False  overrides_differ=True/False
      → _has_overrides                   (per layer)
      → _overrides_differ                (per layer, only if _has_overrides=True)
      → _create_layer_config             (per layer that needs rebuild)
      → _resolve_layer_class             (per layer that needs rebuild)
      → _unregister_layer                (stale context cleanup, per rebuilt layer)
      → _instantiate_layer               (per rebuilt layer)
      → _apply_no_ops                    (per layer, applies NoOpAttention/NoOpMLP/NoOpNorm)
      → _unregister_layer                (per no-op module, removes from static_forward_context)
  ← _patch_anymodel_layers
← AnyModel.__init__
```

### Weight Loading

```
→ AnyModel.load_weights
  → _collect_noop_prefixes           (builds set of weight-name prefixes to skip)
  ← _collect_noop_prefixes           (logs count of prefixes collected)
  → _expand_noop_prefixes_for_mapper (adds HF-style name equivalents)
  ← _expand_noop_prefixes_for_mapper (logs before/after prefix counts)
  (filters weight iterator to drop noop prefixes)
  → super().load_weights             (base model loads remaining weights)
```

### Optional path: `resolve_wrapper_cls`

Called by the model registry before instantiation (e.g. for type checking):

```
→ resolve_wrapper_cls
  → _resolve_arch
  → _get_or_create_wrapper
← resolve_wrapper_cls
```

## Function Reference

| Function | Phase | Purpose |
|---|---|---|
| `AnyModel.__new__` | init | Selects/creates the concrete wrapper subclass |
| `AnyModel.__init__` | init | Runs base model init then patches layers |
| `_resolve_arch` | init | Reads `base_architecture` from hf_config, looks up ArchInfo |
| `_arch_info_from_config` | init | Loads ArchInfo from `hf_config.anymodel_arch_info` if present |
| `_validate_config_arch_info` | init | Security: ensures module paths stay within `vllm.model_executor.models` |
| `_get_or_create_wrapper` | init | Cache lookup or creation of `AnyModel{Arch}` wrapper class |
| `_make_wrapper_cls` | init | Dynamically creates `class AnyModel{Arch}(AnyModel, BaseCls)` |
| `_patch_anymodel_layers` | init | Iterates layers, rebuilds overridden ones, injects no-ops |
| `_has_overrides` | init (per layer) | True if block_config contains any override keys |
| `_overrides_differ` | init (per layer) | True if override values differ from global config |
| `_create_layer_config` | init (per layer) | Deep-copies global config and applies per-layer overrides |
| `_resolve_layer_class` | init (per layer) | Returns the correct decoder layer class (hybrid-aware) |
| `_instantiate_layer` | init (per layer) | Constructs a new layer with per-layer config |
| `_unregister_layer` | init (per layer) | Removes stale entries from `static_forward_context` |
| `_apply_no_ops` | init (per layer) | Replaces attn/ffn/norm sub-modules with identity pass-throughs |
| `AnyModel.load_weights` | weight load | Filters noop weights then delegates to base class |
| `_collect_noop_prefixes` | weight load | Builds frozenset of weight-name prefixes for no-op modules |
| `_expand_noop_prefixes_for_mapper` | weight load | Adds HF-style prefix equivalents for models with `hf_to_vllm_mapper` |
| `resolve_wrapper_cls` | registry | Returns wrapper class without constructing an instance |
