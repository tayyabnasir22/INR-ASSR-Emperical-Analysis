
# INR-ASSR: Empirical Analysis

A centralized repository for **training** and **evaluating** existing **INR-based ASSR** implementations.  
This framework supports multiple **training recipes**, **IQA metrics**, **datasets**, and **encoders**, providing a unified interface for experimentation and benchmarking.


## Getting Started

### Training

    1. Open train.py.  
    2. Uncomment the desired **training strategy**.  
    3. Run the script to begin model training.

```bash
python train.py
```

### Testing

    1. Open test.py.
    2. Uncomment the desired testing strategy.
    3. Run the script to evaluate your trained model.

```bash
python test.py
```

## Configurations

The repository includes several configurable components to enable flexible experimentation. Current configurations reproduce results from existing INR-based ASSR methods, while also allowing easy extension for new:

    - Training and evaluation strategies
    - Datasets
    - INR-based encoders and decoders

Unlike many frameworks, this repository embeds configurations directly within the Python modules rather than relying on external config files, making experimentation and customization more intuitive.



## Training Orchestrator
The **Training Orchestrator** serves as the foundation for managing training across different configurations.  
For any given **encoder** and **decoder**, it supports multiple **training recipes**, each defining a unique combination of loss functions, patch sizes, scale ranges, and learning rate schedulers.

### Available Training Recipes

#### `TrainSimple`
- **Loss:** L1 Loss  
- **Input Patch Size:** 48×48  
- **Scale Range:** 1–4  
- **Learning Rate Scheduler:** Multi-step  

A baseline configuration for standard INR-ASSR training.

---

#### `TrainSimpleLargerPatch`
- **Loss:** L1 Loss  
- **Input Patch Size:** 64×64  
- **Scale Range:** 1–4  
- **Learning Rate Scheduler:** Multi-step  

Trains with larger input patches to enhance spatial consistency.

---

#### `TrainSimpleLargerPatchScale`
- **Loss:** L1 Loss  
- **Input Patch Size:** 64×64  
- **Scale Range:** 1–6  
- **Learning Rate Scheduler:** Multi-step  

Extends `TrainSimpleLargerPatch` by covering a broader range of scales.

---

#### `TrainSGDR`
- **Loss:** L1 Loss  
- **Input Patch Size:** 48×48  
- **Scale Range:** 1–4  
- **Learning Rate Scheduler:** Cosine Annealing (SGDR)  

Uses cosine annealing to enable smoother learning rate transitions.

---

#### `TrainGradLoss`
- **Loss:** Gradient L1 Loss  
- **Input Patch Size:** 48×48  
- **Scale Range:** 1–4  
- **Learning Rate Scheduler:** Multi-step  

Incorporates gradient-based supervision to encourage structural and texture fidelity.

---

#### `TrainGramL1Loss`
- **Loss:** Gram L1 Loss  
- **Input Patch Size:** 48×48  
- **Scale Range:** 1–4  
- **Learning Rate Scheduler:** Multi-step  

Introduces Gram-based loss for texture consistency and perceptual enhancement.

---

### Adding New Training Recipes

Creating new training strategies is straightforward:

    1. Duplicate an existing recipe configuration.  
    2. Adjust the parameters including loss function, patch size, scale range, or learning rate scheduler to match your experimental setup.  
    3. Register the new configuration in the TrainingOrchestrator.py.



## Testing Orchestrator

The **Testing Orchestrator** is the foundation for evaluating models using different testing strategies. 
It provides flexible configuration options to support a variety of datasets, scales, and model architectures, while maintaining sensible defaults that cover most use cases.

By default, all **Testing Orchestrators** are designed to evaluate models across **multiple scales** for a given **encoder** and **decoder**.  
The testing logic and parameters are primarily managed by the **`BaseTestingConfiguration`** module, which defines how models are initialized, loaded, and validated.

Each orchestrator function accepts the following input:
- **`base_test`** — A configuration object containing model (encoder-decoder), training details, testing parameters, and validation options.

This modular design ensures that the same testing framework can be reused across different model architectures and datasets, while maintaining consistent evaluation protocols.

---

### BaseTestingConfiguration

The **`BaseTestingConfiguration`** (located in `Models.BaseTestingConfiguration`) defines and manages all **testing-related configurations**.  
It provides default values suitable for most datasets, while dataset-specific parameters are dynamically updated inside each orchestrator.

This configuration object centralizes all testing parameters, ensuring reproducible evaluation behavior across datasets and models.

---

### Configuration Parameters

| Parameter | Type / Reference | Description |
|------------|------------------|--------------|
| **`encoder`** | `Models.EncoderType` | Name of the encoder architecture used for testing. |
| **`decoder`** | `Models.DecoderType` | Name of the decoder architecture being evaluated. |
| **`recipe`** | `Models.RecipeType` | Name of the training recipe associated with the model. |
| **`input_patch`** | `tuple[int, int]` | Input patch size used during training and testing. |
| **`scale_range`** | `list[int]` | The scale factors used for model training. |
| **`eval_scale`** | `int` | Current scale factor being tested. The orchestrator iterates over multiple scales and updates this dynamically. |
| **`benchmark`** | `Models.BenchmarkType` | Type of dataset used for evaluation (determines whether RGB or Y-channel evaluation is applied). |
| **`valid_data_path`** | `str` | Path to the high-resolution (HR) dataset images. |
| **`valid_data_pathScale`** | `str` or `None` | Path to the corresponding low-resolution (LR) images. If `None`, the code auto-generates bicubic downsampled inputs from HR images. |
| **`total_example`** | `int` | Total number of examples in the dataset (used for logging). |
| **`eval_batch_size`** | `int` | Number of coordinate samples fed to the model at once during evaluation. |
| **`model_name`** | `Models.SavedModelType` | Specifies which model checkpoint to evaluate — e.g., the best PSNR model or the final epoch model. |
| **`test_strategy`** | `TestingStrategy` | Defines whether to test full images or split them into patches for memory-efficient evaluation on smaller GPUs. |
| **`breakdown_patch_size`** | `tuple[int, int]` or `None` | Patch size to use if patch-based testing is enabled. Leave as `None` otherwise. |
| **`overlap`** | `int` or `None` | Overlap size (in pixels) between patches when using overlapping-patch-based testing. Leave as `None` if not used. |

---

### Notes

- The orchestrators automatically **update dataset-specific fields** such as `valid_data_path`, `valid_data_pathScale`, and `eval_scale` based on the dataset and scale being tested.  
- You can **customize or extend** this configuration to include new parameters for specialized testing workflows or datasets.  
- The design ensures that **training, and evaluation pipelines remain decoupled yet fully configurable** through this unified configuration class.



## Saving and Loading Models

The **model saving and loading mechanism** automatically constructs model paths based on the configuration parameters used during training.  
This ensures that each model checkpoint is uniquely identified by its encoder, decoder, recipe, and other settings allowing for seamless testing and reproducibility.

---

### How It Works

Each training run saves its model weights under a directory name that encodes the full configuration.  
When performing evaluation, the **same configuration** must be used to correctly locate and load the desired checkpoint.

Make sure to **use the exact same settings** (encoder, decoder, recipe, etc.) that were used during training when testing or resuming training.

---

### Customization Options

The saving/loading system is fully configurable.  
You can easily modify parameters to:

- Load **specific epoch checkpoints**
- Switch between **different testing strategies** (e.g., full image, patch-based, or overlapped patch testing, to handle **GPU memory constraints** using smaller patch sizes).

The logic for constructing save paths can be customized in  
**`Utilities.PathManager.GetModelSavePath`**.

---

### Default Save Path Format

By default, model checkpoints are saved and loaded using the following path pattern: 

```./model_states_{encoder}_{decoder}_{recipe}_Patch_{input-patch}_Scale_{scale-range-start}_{scale-range-end}/{save-type}.pth```


Where each placeholder in `{braces}` represents a configurable value:

| Variable | Description |
|-----------|--------------|
| **`{encoder}`** | Name of the encoder (`Models.EncoderType`) |
| **`{decoder}`** | Name of the decoder (`Models.DecoderType`) |
| **`{recipe}`** | Name of the training recipe (`Models.RecipeType`) |
| **`{input-patch}`** | Input patch size used during training |
| **`{scale-range-start}`** | First value in the scale range used for training |
| **`{scale-range-end}`** | Last value in the scale range used for training |
| **`{save-type}`** | Type of saved model — e.g., checkpoint from `Models.SavedModelType` (`best`, `last`) or a specific epoch number |

---




