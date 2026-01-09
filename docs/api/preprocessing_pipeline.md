# preprocessing_pipeline

```{eval-rst}
.. automodule:: xpectrass.preprocessing_pipeline
   :members:
   :undoc-members:
   :show-inheritance:
```

## Classes

### FTIRPreprocessor

```{eval-rst}
.. autoclass:: xpectrass.preprocessing_pipeline.FTIRPreprocessor
   :members:
   :undoc-members:
   :show-inheritance:
```

### PreprocessingConfig

```{eval-rst}
.. autoclass:: xpectrass.preprocessing_pipeline.PreprocessingConfig
   :members:
   :undoc-members:
```

## Functions

### create_preprocessor

```python
create_preprocessor(preset: str = 'standard', **overrides) -> FTIRPreprocessor
```

Create preprocessor from preset with optional overrides.

**Parameters:**
- `preset` (str): Preset name ('minimal', 'standard', 'classification', 'pca', 'raw')
- `**overrides`: Override specific configuration options

**Returns:**
- `FTIRPreprocessor`: Configured preprocessor

**Example:**
```python
pipe = create_preprocessor('standard', derivatives=True)
```

### get_preset_config

```python
get_preset_config(name: str) -> PreprocessingConfig
```

Get predefined preprocessing configuration.

**Parameters:**
- `name` (str): Preset name

**Returns:**
- `PreprocessingConfig`: Configuration dataclass
