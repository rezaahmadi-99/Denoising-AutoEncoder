# Image Denoising with a Convolutional Autoencoder

A PyTorch experiment that reconstructs clean grayscale images from inputs corrupted by Gaussian noise. The project uses a small subset of Tiny ImageNet and explores how kernel size, feature channels, and training duration affect reconstruction quality.

The implementation, training history, loss curves, and example reconstructions are in [notebook.ipynb](notebook.ipynb).

## Data and preprocessing

- Dataset: [Tiny ImageNet on Hugging Face](https://huggingface.co/datasets/zh-plus/tiny-imagenet).
- Training: 100 images from the `train` split, selected after shuffling with seed `43`.
- Validation: 100 images from the `valid` split, selected after shuffling with seed `42`.
- Images are converted to grayscale and represented as `1 × 64 × 64` tensors with clean pixel values in `[0, 1]`.
- Independent Gaussian noise is generated for training and validation, with mean `0` and standard deviation `0.05`.
- Both noise tensors remain fixed throughout a training run. Noisy inputs are not clipped to `[0, 1]`.

The executable code uses noise standard deviation **0.05**; an older notebook markdown cell still mentions `0.2`.

## Model

The encoder uses three convolutions, and the decoder uses three transposed convolutions. Every layer uses a `3 × 3` kernel, stride `1`, and no padding.

| Stage | Operation | Output shape, excluding batch |
| --- | --- | --- |
| Input | Noisy grayscale image | `1 × 64 × 64` |
| Encoder 1 | Conv + BatchNorm + ReLU | `16 × 62 × 62` |
| Encoder 2 | Conv + BatchNorm + ReLU | `16 × 60 × 60` |
| Encoder 3 | Conv + ReLU | `4 × 58 × 58` |
| Decoder 1 | Transposed Conv + BatchNorm + ReLU | `16 × 60 × 60` |
| Decoder 2 | Transposed Conv + BatchNorm + ReLU | `16 × 62 × 62` |
| Decoder 3 | Transposed Conv + ReLU | `1 × 64 × 64` |

The model predicts the clean image directly. Its final ReLU makes outputs nonnegative but does not impose an upper bound of `1`. Although spatial dimensions shrink in the encoder, the latent tensor contains more activations than the input, so this is not a compact image-compression model.

## Setup

The notebook was developed using Python 3.13. From the project directory, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy pandas matplotlib seaborn pillow datasets jupyterlab ipykernel
```

On Windows, activate it with `.venv\Scripts\activate` instead.

Dependencies are not currently pinned to exact versions. The first dataset load requires internet access and can download substantially more data than the selected 100-image subsets: selection happens after loading the dataset splits.

## Run the notebook

Open `notebook.ipynb` in VS Code with the Python and Jupyter extensions, select the `.venv` kernel, and run the cells from top to bottom. Alternatively, launch JupyterLab:

```bash
python -m jupyterlab notebook.ipynb
```

Use notebook cell execution, rather than `python notebook.ipynb`: an `.ipynb` file is a JSON document, not a Python script.

The notebook runs on CPU by default. Its workflow is:

1. Load and select the training and validation images.
2. Convert images to tensors and generate noisy inputs.
3. Define the encoder, decoder, and combined autoencoder.
4. Construct paired datasets and the training data loader.
5. Train the model and evaluate all 100 validation images after each epoch.
6. Plot training/validation MSE and compare an original, noisy, and reconstructed image.

## Training configuration

| Setting | Value |
| --- | --- |
| Loss | Mean squared error (`nn.MSELoss`) |
| Optimizer | Adam |
| Learning rate | `0.001` |
| Epochs | `200` |
| Training batch size | `20` |
| Validation | Entire validation subset, every epoch |

Training uses `model.train()`. Validation uses `model.eval()` and `torch.no_grad()`. The training loss is averaged over five equal-sized batches; validation MSE is computed over the full validation tensor.

Rerunning the training cell creates a new model and optimizer. To continue an existing run, preserve those objects. If noisy tensors are regenerated, rerun the dataset/data-loader cell too: existing `TensorDataset` objects retain references to the previous tensors.

## Results and interpretation

The currently saved notebook run reports a final training MSE of approximately **0.01171** at epoch 200. Validation losses are plotted in the notebook, but their numerical values are not printed in this version. Model initialization, noise generation, and batch shuffling are not fully seeded, so reruns can produce different results.

For noise with standard deviation `0.05`, returning the noisy input unchanged has an expected MSE of `0.05² = 0.0025`. Lower training loss and smoother-looking images alone do not establish successful denoising: evaluate whether reconstruction error beats the actual noisy-input error on validation data.

After training, this cell compares both quantities using the tensors actually held by the validation dataset:

```python
model.eval()
with t.no_grad():
    clean, noisy = valid_tensor_pairs[:]
    reconstructed = model(noisy)
    baseline_mse = F.mse_loss(noisy, clean).item()
    reconstruction_mse = F.mse_loss(reconstructed, clean).item()

print(f"Noisy-input MSE:    {baseline_mse:.6f}")
print(f"Reconstruction MSE: {reconstruction_mse:.6f}")
```

For visual comparisons, use `cmap="gray", vmin=0, vmax=1` in every `imshow` call. The current plots scale each panel independently, which can conceal brightness differences.

The current experiment uses only 100 training images and 100 validation images. Validation is used to guide architecture choices; a separate untouched test set is needed for a final performance assessment. Trained weights are not automatically saved by the notebook.

## Possible extensions

- Predict noise and subtract it from the input using residual learning.
- Add matching encoder-to-decoder skip connections to preserve detail.
- Train on more varied images or patches, with fresh training noise per batch.
- Save the checkpoint with the lowest validation loss.
- Report PSNR and inspect multiple fixed examples alongside MSE.

## Related papers

- [DnCNN: Beyond a Gaussian Denoiser—Residual Learning of Deep CNN for Image Denoising](https://arxiv.org/abs/1608.03981) — residual noise prediction; [author's code](https://github.com/cszn/DnCNN).
- [Image Restoration Using Very Deep Convolutional Encoder–Decoder Networks with Symmetric Skip Connections](https://arxiv.org/abs/1603.09056) — encoder–decoder skip connections.
- [FFDNet: Toward a Fast and Flexible Solution for CNN-Based Image Denoising](https://arxiv.org/abs/1710.04026) — conditioning on noise level; [author's code](https://github.com/cszn/FFDNet).

These are references for future experiments; the current notebook does not reproduce their full architectures or training protocols.
