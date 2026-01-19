# SimCLR Self-Supervised Learning - Discussion & Implementation Plan

**Date**: 2025-12-04
**Project**: CerraData Phenorob - Self-Supervised Pretraining Experiment

---

## Context: Your Experiments So Far

You've been running comparison experiments on semantic segmentation:

| Experiment | Pretraining | Fine-tuning | Status |
|------------|-------------|-------------|--------|
| **Experiment 1** | None (random init) | 5% L2 labels | ✅ Complete |
| **Experiment 2** | 100% L1 labels (7 classes) | 5% L2 labels (14 classes) | ✅ Complete |
| **Experiment 3** | Self-supervised (SimCLR) | 5% L2 labels | 🔄 Planning |

**Goal**: Compare self-supervised pretraining (using unlabeled data) vs supervised pretraining (L1 labels) vs no pretraining.

---

## What is Self-Supervised Learning?

Self-supervised learning trains models **without labels** by creating a "pretext task" from the data itself.

**Key Idea**: Instead of predicting labels, predict relationships between augmented versions of the same image.

### Analogy:
- **Supervised**: "This is a wheat field" (requires label)
- **Self-supervised**: "These two crops are from the same field" (no label needed!)

---

## What is Contrastive Learning?

Contrastive learning is a type of self-supervised learning that works by:

1. **Creating pairs**: Take an image, create 2 augmented versions (different crops, rotations, etc.)
2. **Learning similarity**: Train the model to recognize that the 2 versions are of the SAME image
3. **Learning dissimilarity**: Make sure the model can distinguish them from DIFFERENT images

**Mathematical formulation**:
- Maximize similarity: sim(augment1(img_i), augment2(img_i)) → high
- Minimize similarity: sim(augment1(img_i), augment(img_j)) → low (where j ≠ i)

---

## Available Contrastive Learning Methods

### 1. **SimCLR** (Simple Contrastive Learning of Representations)
**Characteristics**:
- ✅ Simplest to understand and implement
- ✅ No momentum encoder, no queue management
- ⚠️ Needs moderate-to-large batch sizes (64-1024)
- ✅ Single encoder architecture

**How it works**:
1. Batch of N images → Create 2 augmented views each → 2N total images
2. Pass all through encoder → Get feature vectors
3. For each image, its 2 views are "positive pair"
4. All other 2(N-1) images are "negative pairs"
5. Loss: InfoNCE/NT-Xent contrastive loss

**Pros**:
- Minimal code
- Easy to debug
- Works well with your batch_size=100

**Cons**:
- Batch size limited negatives (198 negatives per sample with N=100)

---

### 2. **MoCo** (Momentum Contrast)
**Characteristics**:
- More complex implementation
- Uses 2 encoders: query + momentum-updated key encoder
- Maintains queue of 65,536 negative samples
- Works with small batch sizes (32-64)

**How it works**:
1. Query encoder: normal training
2. Key encoder: momentum-updated copy (slowly follows query)
3. Queue stores past encoded features as negatives
4. More negatives = better representations

**Pros**:
- Works with tiny batches
- More negatives (65k vs 198)
- Memory efficient

**Cons**:
- More complex (momentum update, queue management)
- Harder to debug

---

### 3. **BYOL** (Bootstrap Your Own Latent)
**Characteristics**:
- No negative pairs needed!
- Uses predictor network + momentum encoder
- More stable than expected (surprising theoretically)

**Pros**:
- No negatives = simpler conceptually
- Works well empirically

**Cons**:
- Can be unstable
- Less intuitive (why does it work without negatives?)

---

### 4. **SimSiam** (Simple Siamese)
**Characteristics**:
- Even simpler than BYOL
- No momentum encoder, just stop-gradient

**Pros**:
- Minimal implementation
- No queue, no momentum

**Cons**:
- Can be unstable
- Less widely used

---

## Recommendation: SimCLR ✅

**Why SimCLR for your project**:
1. ✅ **Simplest baseline** - perfect for understanding/learning
2. ✅ **Minimal code** - follows your existing patterns
3. ✅ **Your batch_size=100 works** - sufficient for agricultural data
4. ✅ **Easy to debug** - single encoder, straightforward loss
5. ✅ **Well-documented** - extensive literature and examples

---

## Technical Deep Dive: How SimCLR Works

### Architecture Overview

```
Input: Minibatch of N images {x₁, x₂, ..., xₙ}

┌─────────────────────────────────────────────┐
│ Step 1: Data Augmentation                   │
│ For each xᵢ, create 2 augmented views:      │
│   - x̃₂ᵢ₋₁ = augment₁(xᵢ)                    │
│   - x̃₂ᵢ   = augment₂(xᵢ)                    │
│ Result: 2N augmented images                 │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ Step 2: Encoder Network f(·)                │
│ Pass all 2N images through ResNet34:        │
│   hᵢ = f(x̃ᵢ) ∈ ℝ⁵¹²                         │
│ (Your U-Net encoder, output 512-dim)        │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ Step 3: Projection Head g(·)                │
│ Apply MLP (512 → 256 → 128):                │
│   zᵢ = g(hᵢ) ∈ ℝ¹²⁸                         │
│ Note: g(·) discarded after pretraining!     │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ Step 4: L2 Normalization                    │
│   z'ᵢ = zᵢ / ||zᵢ||                          │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ Step 5: NT-Xent Contrastive Loss            │
│ (Details below)                              │
└─────────────────────────────────────────────┘
```

---

### The NT-Xent Loss (Normalized Temperature-scaled Cross Entropy)

**This is the heart of SimCLR.**

#### Mathematical Formulation:

For a positive pair (i, j) where j is the other augmentation of image i:

```
ℓ(i,j) = -log[ exp(sim(zᵢ, zⱼ)/τ) / Σₖ₌₁²ᴺ 𝟙[k≠i] · exp(sim(zᵢ, zₖ)/τ) ]

Where:
- sim(u, v) = (u · v) / (||u|| ||v||)  [cosine similarity]
- τ = temperature parameter (hyperparameter, typically 0.5)
- 𝟙[k≠i] = indicator function (1 if k≠i, else 0)
- Denominator sums over all 2N-1 other images (excluding i itself)
```

#### Intuitive Breakdown:

```
Numerator: exp(sim(zᵢ, zⱼ)/τ)
- High if zᵢ and zⱼ are similar (positive pair)
- Gradient pulls zᵢ and zⱼ closer

Denominator: Σₖ exp(sim(zᵢ, zₖ)/τ)
- Sum over ALL samples (positive + negatives)
- Large if zᵢ is similar to many samples
- Gradient pushes zᵢ away from negatives

Loss minimization:
- Maximize numerator (pull positives together)
- Minimize denominator (push negatives apart)
```

#### Temperature τ:

```
τ = 0.1 (low):
  - Very spiky distribution
  - Model must be very precise
  - Harder optimization

τ = 0.5 (medium, default):
  - Balanced concentration
  - Good optimization dynamics
  - Standard choice ✅

τ = 1.0 (high):
  - Smoother distribution
  - More forgiving
  - Potentially less discriminative
```

---

### Why Projection Head g(·)?

**Key empirical finding** from SimCLR paper (Chen et al., 2020):

| Setup | Pretrain | Transfer | Result |
|-------|----------|----------|--------|
| With projection | z = g(h) | Use h (discard g) | ✅ Best performance |
| With projection | z = g(h) | Use z | ❌ Worse |
| No projection | h only | Use h | ❌ Much worse |

**Why does this work?**

1. **Information loss**: Contrastive loss discards some information (only cares about similarity, not absolute values)
2. **Separation of concerns**:
   - Projection head g learns task-specific similarity (augmentation invariance)
   - Encoder h retains general semantic information
3. **Bottleneck effect**: g acts as a bottleneck that removes augmentation-specific artifacts

**Result**: We train with g, but transfer h to downstream tasks!

---

### Gradient Flow Analysis

Taking derivative of loss w.r.t. zᵢ:

```
∂ℓ/∂zᵢ = -(1/τ) · [zⱼ - Σₖ p(k|i) · zₖ]

where p(k|i) = exp(sim(zᵢ, zₖ)/τ) / Σₖ exp(sim(zᵢ, zₖ)/τ)
```

**Interpretation**:
- **Attractive force**: zⱼ pulls zᵢ (positive sample)
- **Repulsive force**: Weighted average Σₖ p(k|i)·zₖ pushes zᵢ (negatives)
- **Hard negative mining**: p(k|i) is larger for more similar negatives
  - Hard negatives (similar but wrong) contribute more to gradient
  - Automatic curriculum learning!

---

### Concrete Example: Batch Size = 4

Let's trace through with 4 images:

```python
# Input
images = [img1, img2, img3, img4]  # Shape: (4, 12, 128, 128)

# Step 1: Augmentation
aug1 = augment(images)  # [img1_v1, img2_v1, img3_v1, img4_v1]
aug2 = augment(images)  # [img1_v2, img2_v2, img3_v2, img4_v2]
batch = concat([aug1, aug2])  # Shape: (8, 12, 128, 128)

# Step 2: Encode
features = encoder(batch)  # Shape: (8, 512)

# Step 3: Project
z = projection_head(features)  # Shape: (8, 128)
z = F.normalize(z, dim=1)  # L2 normalize

# Step 4: Similarity matrix
sim_matrix = z @ z.T  # Shape: (8, 8)

# sim_matrix:
#      0    1    2    3    4    5    6    7
# 0 [1.0  0.1  0.1  0.1  0.9  0.1  0.1  0.1]  ← img1_v1
# 1 [0.1  1.0  0.1  0.1  0.1  0.9  0.1  0.1]  ← img2_v1
# 2 [0.1  0.1  1.0  0.1  0.1  0.1  0.9  0.1]  ← img3_v1
# 3 [0.1  0.1  0.1  1.0  0.1  0.1  0.1  0.9]  ← img4_v1
# 4 [0.9  0.1  0.1  0.1  1.0  0.1  0.1  0.1]  ← img1_v2
# 5 [0.1  0.9  0.1  0.1  0.1  1.0  0.1  0.1]  ← img2_v2
# 6 [0.1  0.1  0.9  0.1  0.1  0.1  1.0  0.1]  ← img3_v2
# 7 [0.1  0.1  0.1  0.9  0.1  0.1  0.1  1.0]  ← img4_v2

# Note: High similarity (0.9) between positive pairs
#       Low similarity (0.1) between negatives

# Step 5: Compute loss
# For img1 (indices 0 and 4):
#   Positive: sim(z₀, z₄) = 0.9
#   Negatives: sim(z₀, [z₁,z₂,z₃,z₅,z₆,z₇])

loss_0 = -log[exp(0.9/0.5) / (exp(0.1/0.5) + ... + exp(0.9/0.5))]

# Repeat for all 8 samples, average
```

---

### Why Does This Work? (The Big Picture)

#### 1. Augmentation Invariance
```
Original image x → encoder → h

Augmented view 1: crop + flip → x̃₁ → encoder → h₁
Augmented view 2: rotate + blur → x̃₂ → encoder → h₂

Loss encourages: sim(h₁, h₂) ≈ 1

Result: Encoder learns to ignore augmentation artifacts
        Focuses on semantic content (what the image represents)
```

#### 2. Instance Discrimination
```
Different images should have different representations:
  img1 → h₁
  img2 → h₂
  Loss encourages: sim(h₁, h₂) ≈ 0

Result: Encoder learns discriminative features
        Can distinguish between different semantic content
```

#### 3. Preventing Collapse
```
Trivial solution: f(x) = constant for all x
- All similarities = 1
- Numerator: exp(1/τ)
- Denominator: exp(1/τ) × (2N-1)
- Loss = -log[1/(2N-1)] = log(2N-1) ≈ HIGH

Negatives prevent collapse by making denominator large
when representations are too similar!
```

#### 4. Transfer Learning
After pretraining, encoder has learned:
- ✅ Spatial invariance (from crops)
- ✅ Rotation invariance (from rotations)
- ✅ Color invariance (from jittering)
- ✅ Semantic features (from discrimination)

For downstream segmentation task:
- Load pretrained encoder weights
- Add decoder head
- Fine-tune on small labeled data (5%)
- Encoder already knows good features!
- Just need to map features → classes

---

## Augmentations for Sentinel-2 Multispectral Data

**Standard Computer Vision Augmentations**:
```python
1. RandomResizedCrop(size=128, scale=(0.5, 1.0))
   - Crop random region, resize to 128x128
   - Forces spatial invariance

2. RandomHorizontalFlip(p=0.5)
   - Flip left-right
   - Agricultural fields look same from both sides

3. RandomVerticalFlip(p=0.5)
   - Flip up-down
   - Satellite imagery has no preferred orientation

4. RandomRotation(degrees=90)
   - Rotate 0°, 90°, 180°, or 270°
   - Geographic invariance

5. GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
   - Blur image
   - Robust to noise/resolution changes
```

**Multispectral-Specific Augmentations**:
```python
6. SpectralNoise
   - Add Gaussian noise to each band independently
   - Simulates sensor noise

7. SpectralScaling
   - Scale each band by random factor [0.8, 1.2]
   - Simulates illumination/atmospheric variations

8. BandDropout
   - Randomly zero out 1-2 bands
   - Forces model to not rely on single band
```

**Strategy**:
- Start with conservative (items 1-5)
- Add spectral augmentations (6-8) if needed
- Stronger augmentation = better invariance (but harder optimization)

---

## Implementation Plan

### Files to Create

#### 1. `simclr_augmentations.py` (~100 lines)
**Purpose**: Data augmentation pipeline

**Components**:
- `SimCLRAugmentation` class
- Compose multiple augmentations
- Handle 12-channel images (not just RGB)

**Key functions**:
```python
class SimCLRAugmentation:
    def __init__(self, image_size=128):
        self.transforms = transforms.Compose([
            RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(degrees=90),
            GaussianBlur(kernel_size=23),
            SpectralNoise(std=0.1),
        ])

    def __call__(self, image):
        return self.transforms(image)
```

---

#### 2. `simclr_dataset.py` (~80 lines)
**Purpose**: Unlabeled dataset loader (just MSI images)

**Key design**:
- Loads MSI images WITHOUT labels (self-supervised!)
- Returns 2 augmented views per image
- Reuses existing data split structure

**Code structure**:
```python
class UnlabeledCerraDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir) / split / "images"
        self.image_files = list(self.data_dir.glob("*.tif"))
        self.transform = transform

    def __getitem__(self, idx):
        # Load MSI image (12 channels)
        image = load_tif(self.image_files[idx])

        # Create 2 augmented views
        view1 = self.transform(image)
        view2 = self.transform(image)  # Different random augmentation

        return view1, view2
```

**Key point**: No labels needed! Uses ALL training images.

---

#### 3. `simclr_model.py` (~150 lines)
**Purpose**: SimCLR model implementation

**Components**:
- `SimCLRPretraining` (PyTorch Lightning module)
- `nt_xent_loss()` function
- Encoder extraction from U-Net
- Projection head

**Code structure**:
```python
class SimCLRPretraining(pl.LightningModule):
    def __init__(self, temperature=0.5, projection_dim=128):
        super().__init__()

        # Extract encoder from U-Net (SAME as your existing model)
        base_unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=12,
            classes=14  # Doesn't matter, we only use encoder
        )
        self.encoder = base_unet.encoder

        # Projection head (512 → 256 → 128)
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

        self.temperature = temperature

    def forward(self, x):
        # Encode
        encoder_features = self.encoder(x)
        h = encoder_features[-1]  # Last layer

        # Global average pooling
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)

        # Project
        z = self.projection_head(h)

        # L2 normalize
        z = F.normalize(z, dim=1)

        return z

    def training_step(self, batch, batch_idx):
        view1, view2 = batch

        # Get representations
        z1 = self(view1)
        z2 = self(view2)

        # Compute contrastive loss
        loss = nt_xent_loss(z1, z2, self.temperature)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def save_encoder(self, path):
        """Save ONLY encoder weights (not projection head)"""
        torch.save(self.encoder.state_dict(), path)


def nt_xent_loss(z1, z2, temperature=0.5):
    """
    NT-Xent loss for contrastive learning

    Args:
        z1: (N, d) - first view embeddings
        z2: (N, d) - second view embeddings
        temperature: scaling factor

    Returns:
        scalar loss
    """
    N = z1.shape[0]

    # Concatenate to (2N, d)
    z = torch.cat([z1, z2], dim=0)

    # Similarity matrix (2N, 2N)
    sim_matrix = torch.mm(z, z.T) / temperature

    # Create positive pair mask
    positive_mask = torch.zeros(2*N, 2*N, dtype=torch.bool, device=z.device)
    for i in range(N):
        positive_mask[i, i+N] = True
        positive_mask[i+N, i] = True

    # Create negative mask (exclude self-similarity)
    negative_mask = torch.ones(2*N, 2*N, dtype=torch.bool, device=z.device)
    negative_mask.fill_diagonal_(False)

    # Compute loss using LogSumExp trick for numerical stability
    loss = 0
    for i in range(2*N):
        # Positive similarity
        pos_sim = sim_matrix[i, positive_mask[i]].squeeze()

        # All similarities (excluding self)
        all_sim = sim_matrix[i, negative_mask[i]]

        # NT-Xent loss
        loss_i = -pos_sim + torch.logsumexp(all_sim, dim=0)
        loss += loss_i

    return loss / (2*N)
```

---

#### 4. `train_simclr.py` (~200 lines)
**Purpose**: Self-supervised pretraining script

**Structure** (mirrors `train_baseline.py`):
```python
def train_simclr(
    data_dir,
    batch_size=100,
    num_epochs=200,
    learning_rate=1e-3,
    temperature=0.5,
    projection_dim=128,
    num_workers=4,
    gpu_ids=None,
    checkpoint_dir="./checkpoints_data_splitted",
    log_dir="./logs_splitted",
    experiment_name="simclr_pretrain"
):
    """Self-supervised pretraining with SimCLR"""

    print("=== SimCLR Self-Supervised Pretraining ===")

    # Create augmentation
    augmentation = SimCLRAugmentation(image_size=128)

    # Create unlabeled dataset (train split only)
    train_dataset = UnlabeledCerraDataset(
        data_dir=data_dir,
        split='train',
        transform=augmentation
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # Create SimCLR model
    model = SimCLRPretraining(
        temperature=temperature,
        projection_dim=projection_dim
    )

    # Setup trainer (like train_baseline.py)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator='gpu' if gpu_ids else 'cpu',
        devices=gpu_ids if gpu_ids else 1,
        logger=TensorBoardLogger(log_dir, name=experiment_name),
        callbacks=[...],
    )

    # Train
    trainer.fit(model, train_loader)

    # Save encoder weights
    encoder_path = f"{checkpoint_dir}/{experiment_name}/encoder_final.pth"
    model.save_encoder(encoder_path)

    print(f"Encoder saved to: {encoder_path}")
    return encoder_path
```

**Command-line arguments**:
- `--data_dir`: Path to dataset
- `--batch_size`: Default 100
- `--num_epochs`: Default 200
- `--temperature`: Default 0.5
- `--projection_dim`: Default 128
- Standard args: gpu_ids, checkpoint_dir, log_dir, etc.

---

#### 5. `train_l2_from_simclr.py` (~180 lines)
**Purpose**: Fine-tune L2 segmentation from SimCLR encoder

**Key differences from `train_l2_finetune.py`**:
- Loads encoder-only weights (not full PyTorch Lightning checkpoint)
- Decoder randomly initialized
- Otherwise identical

**Code structure**:
```python
def train_l2_from_simclr(
    simclr_encoder_path,
    data_dir,
    batch_size=100,
    num_epochs=100,
    learning_rate=1e-4,  # Lower for fine-tuning
    data_percentage=5,
    ...
):
    """Fine-tune L2 model from SimCLR pretrained encoder"""

    print("=== L2 Fine-tuning from SimCLR ===")
    print(f"SimCLR encoder: {simclr_encoder_path}")

    # Create L2 data loaders (5% labeled data)
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        label_level='L2',
        data_percentage=data_percentage  # 5%
    )

    # Create L2 model
    l2_model = UNetSegmentation(
        in_channels=12,
        num_classes=14,
        encoder_name="resnet34",
        learning_rate=learning_rate
    )

    # Load SimCLR pretrained encoder weights
    print("Loading SimCLR pretrained encoder...")
    encoder_state_dict = torch.load(simclr_encoder_path)
    l2_model.model.encoder.load_state_dict(encoder_state_dict)
    print("✓ Encoder weights loaded")

    # Decoder is randomly initialized (no pretraining)
    print("Decoder randomly initialized")

    # Train on 5% L2 labeled data
    trainer = pl.Trainer(...)
    trainer.fit(l2_model, train_loader, val_loader)

    # Test on full test set
    trainer.test(l2_model, test_loader, ckpt_path='best')

    return l2_model
```

---

#### 6. `run_simclr_pretrain.sh` (~20 lines)
**Purpose**: SLURM job script for SimCLR pretraining

```bash
#!/bin/bash
#SBATCH --job-name=simclr_pretrain
#SBATCH --output=logs_splitted/simclr_pretrain_%j.out
#SBATCH --error=logs_splitted/simclr_pretrain_%j.err

echo "=== SIMCLR SELF-SUPERVISED PRETRAINING ==="
echo "Starting at: $(date)"

python train_simclr.py \
    --data_dir /home/s52melba/CerraData_Project_Phenorob/cerradata_splitted \
    --experiment_name "simclr_pretrain_resnet34" \
    --gpu_ids "0" \
    --batch_size 100 \
    --num_epochs 200 \
    --learning_rate 1e-3 \
    --temperature 0.5 \
    --projection_dim 128 \
    --checkpoint_dir ./checkpoints_data_splitted \
    --log_dir ./logs_splitted

echo "Completed at: $(date)"
```

---

#### 7. `run_l2_from_simclr.sh` (~25 lines)
**Purpose**: SLURM job script for L2 fine-tuning

```bash
#!/bin/bash
#SBATCH --job-name=l2_from_simclr
#SBATCH --output=logs_splitted/l2_from_simclr_%j.out
#SBATCH --error=logs_splitted/l2_from_simclr_%j.err

# UPDATE THIS PATH after SimCLR pretraining completes!
SIMCLR_ENCODER="./checkpoints_data_splitted/simclr_pretrain_resnet34_TIMESTAMP/encoder_final.pth"

echo "=== L2 FINE-TUNING FROM SIMCLR PRETRAINING ==="
echo "Using SimCLR encoder: $SIMCLR_ENCODER"
echo "Starting at: $(date)"

python train_l2_from_simclr.py \
    --simclr_encoder "$SIMCLR_ENCODER" \
    --data_dir /home/s52melba/CerraData_Project_Phenorob/cerradata_splitted \
    --experiment_name "l2_finetune_from_simclr_5percent" \
    --gpu_ids "0" \
    --batch_size 100 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --data_percentage 5 \
    --checkpoint_dir ./checkpoints_data_splitted \
    --log_dir ./logs_splitted

echo "Completed at: $(date)"
```

---

## Workflow: Running the Experiment

### Phase 1: Self-Supervised Pretraining
```bash
# Submit SimCLR pretraining job
sbatch run_simclr_pretrain.sh

# Monitor progress
tail -f logs_splitted/simclr_pretrain_*.out

# What to look for:
# - Contrastive loss decreasing (starts ~log(200)≈5.3, ends ~3-4)
# - Stable training (no NaN, no divergence)
# - ~4-8 hours for 200 epochs
```

**Expected output**:
- Checkpoint: `checkpoints_data_splitted/simclr_pretrain_resnet34_TIMESTAMP/`
- Encoder weights: `encoder_final.pth`
- TensorBoard logs: `logs_splitted/simclr_pretrain_resnet34/`

---

### Phase 2: Fine-tune on L2 Labels
```bash
# 1. Update SIMCLR_ENCODER path in run_l2_from_simclr.sh
#    Replace TIMESTAMP with actual timestamp from Phase 1

# 2. Submit fine-tuning job
sbatch run_l2_from_simclr.sh

# 3. Monitor progress
tail -f logs_splitted/l2_from_simclr_*.out

# What to look for:
# - Validation F1/IoU increasing
# - Better performance than baseline?
# - Comparable to L1→L2 hierarchical?
```

**Expected output**:
- Checkpoint: `checkpoints_data_splitted/l2_finetune_from_simclr_5percent_TIMESTAMP/`
- Best model: `best.ckpt`
- Test metrics in training summary

---

## Comparison Framework

After running all experiments, compare:

| Experiment | Pretraining | Pretraining Data | Fine-tuning Data | Test F1 (macro) | Test IoU (macro) |
|------------|-------------|------------------|------------------|-----------------|------------------|
| **Baseline** | None | - | 5% L2 | ??? | ??? |
| **Hierarchical** | L1 supervised | 100% L1 labels | 5% L2 | ??? | ??? |
| **SimCLR** | Self-supervised | 100% unlabeled MSI | 5% L2 | ??? | ??? |

**Hypothesis**:
- SimCLR > Baseline (pretraining helps!)
- SimCLR ≈ or > Hierarchical (self-supervised can match/exceed supervised)

**Why SimCLR might win**:
- L1 labels constrain representations to 7 classes
- SimCLR learns unconstrained features (more general)
- Better transfer to 14-class L2 task

---

## Hyperparameters

### SimCLR Pretraining:
```python
batch_size = 100          # Your current setting
num_epochs = 200          # Start with 200, extend if needed
learning_rate = 1e-3      # Same as baseline training
temperature = 0.5         # Standard for SimCLR
projection_dim = 128      # Standard
optimizer = Adam
scheduler = ReduceLROnPlateau(patience=10)
```

### L2 Fine-tuning:
```python
batch_size = 100
num_epochs = 100
learning_rate = 1e-4      # Lower for fine-tuning (10x less)
data_percentage = 5       # Use 5% labeled data
optimizer = Adam
scheduler = ReduceLROnPlateau(patience=5)
```

---

## Expected Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| **Implementation** | 2-3 hours | Write 7 new files (~755 lines code) |
| **SimCLR pretraining** | 4-8 hours | 200 epochs on full train set |
| **L2 fine-tuning** | 2-4 hours | 100 epochs on 5% labeled data |
| **Analysis** | 1-2 hours | Compare metrics, generate plots |
| **Total** | ~9-17 hours | Full experiment end-to-end |

---

## Key Design Decisions

### 1. **Encoder Compatibility** ✅
- Extract encoder from `smp.Unet` (identical to your existing model)
- Ensures 100% weight compatibility for fine-tuning
- No architectural modifications needed

### 2. **Minimal Code** ✅
- Reuse PyTorch Lightning patterns from existing code
- Follow same structure as `train_baseline.py` and `train_l2_finetune.py`
- Easy to understand and debug

### 3. **Projection Head** ✅
- Used ONLY during pretraining
- Discarded after (only encoder weights transferred)
- Standard SimCLR practice

### 4. **Augmentation Strategy** ✅
- Start conservative (crop, flip, rotate, blur)
- Add spectral augmentations if needed
- Stronger than supervised learning augmentations

### 5. **No Validation During Pretraining** ✅
- Self-supervised loss is the metric
- No ground truth to validate against
- Monitor contrastive loss convergence

---

## Success Metrics

### During SimCLR Pretraining:
- ✅ Contrastive loss decreases steadily
- ✅ Final loss: typically 3-5 (never reaches 0, that's okay!)
- ✅ No NaN or divergence
- ✅ Training stable throughout

### During L2 Fine-tuning:
- ✅ Validation F1/IoU increase
- ✅ Better than baseline (no pretraining)
- ✅ Comparable or better than hierarchical (L1→L2)

### Final Evaluation:
- ✅ Test on full test set (100%)
- ✅ Per-class F1 and IoU scores
- ✅ Confusion matrix analysis
- ✅ Qualitative segmentation visualizations

---

## Troubleshooting

### Problem: Loss not decreasing
**Possible causes**:
- Temperature too low/high → Try adjusting τ
- Augmentations too strong → Reduce augmentation strength
- Learning rate too high/low → Try 1e-4 or 5e-4
- Batch size too small → Increase if possible

### Problem: NaN loss
**Possible causes**:
- Numerical instability in exp() → Use log-sum-exp trick (implemented)
- Learning rate too high → Reduce to 1e-4
- Batch normalization issues → Check normalization

### Problem: Encoder doesn't transfer well
**Possible causes**:
- Not enough pretraining epochs → Train longer (400 epochs)
- Augmentations not relevant → Adjust augmentation strategy
- Temperature not optimal → Try τ=0.1 or τ=1.0

---

## Alternative: Pivot to MoCo if Needed

If SimCLR doesn't work well (unlikely with batch_size=100):

**When to pivot**:
- Batch size constraints (< 64)
- Poor performance despite tuning
- Want more negatives

**MoCo changes**:
- Add momentum encoder (slow EMA of query encoder)
- Add queue (store 65,536 past features)
- More complex but works with tiny batches

**Implementation effort**: +300 lines of code

---

## Files Summary

### New Files (7 total):
1. `simclr_augmentations.py` - Augmentation pipeline
2. `simclr_dataset.py` - Unlabeled data loader
3. `simclr_model.py` - SimCLR model + NT-Xent loss
4. `train_simclr.py` - Pretraining script
5. `train_l2_from_simclr.py` - Fine-tuning script
6. `run_simclr_pretrain.sh` - SLURM pretraining job
7. `run_l2_from_simclr.sh` - SLURM fine-tuning job

### Modified Files:
**None!** All existing code unchanged.

### Total Code:
~755 lines (following your existing patterns)

---

## Questions / Decisions

Before implementation, confirm:

1. **Number of pretraining epochs**:
   - [ ] Start with 200 epochs (~4-6 hours) ← **Recommended**
   - [ ] Go for 400 epochs (~8-12 hours)

2. **Augmentation strategy**:
   - [ ] Conservative (crop, flip, rotate, blur) ← **Recommended to start**
   - [ ] Aggressive (add spectral noise, scaling)

3. **Temperature**:
   - [ ] τ = 0.5 (standard) ← **Recommended**
   - [ ] Try different value?

4. **Projection dimension**:
   - [ ] 128-dim (standard) ← **Recommended**
   - [ ] 256-dim (less compression)

**All have sensible defaults - can proceed with recommendations!**

---

## References

**SimCLR Paper**:
- Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020
- https://arxiv.org/abs/2002.05709

**Key Insights**:
- Projection head improves performance
- Temperature scaling is critical
- Composition of augmentations matters
- Batch size affects quality (but 100 is reasonable)

**MoCo Paper** (alternative):
- He et al., "Momentum Contrast for Unsupervised Visual Representation Learning", CVPR 2020
- https://arxiv.org/abs/1911.05722

---

## Next Steps

1. ✅ **Read and understand this document** (you are here!)
2. ⬜ **Confirm hyperparameters** (use defaults or customize?)
3. ⬜ **Implement 7 files** (~2-3 hours)
4. ⬜ **Run SimCLR pretraining** (submit SLURM job)
5. ⬜ **Monitor training** (check logs, TensorBoard)
6. ⬜ **Run L2 fine-tuning** (submit SLURM job)
7. ⬜ **Compare results** (baseline vs hierarchical vs SimCLR)
8. ⬜ **Analyze and report** (which pretraining strategy wins?)

---

## How to Continue This Conversation

When you come back to this later, tell Claude Code:

```
"I want to implement SimCLR self-supervised learning for my
CerraData project. I have the plan documented in
SIMCLR_DISCUSSION_AND_PLAN.md. Please read that file and
start implementing the 7 files we discussed."
```

Claude Code will:
1. Read this document
2. Understand the full context
3. Start implementing following your existing code patterns
4. Create all 7 files with proper integration

**Alternative**: Just say "Continue with SimCLR implementation" and reference this file!

---

## Summary (TL;DR)

**Goal**: Pretrain U-Net encoder with self-supervised learning (no labels needed)

**Method**: SimCLR contrastive learning
- Create 2 augmented views per image
- Train encoder to recognize same image despite augmentations
- Learn semantic features without labels

**Implementation**: 7 new files (~755 lines)
- Augmentation pipeline
- Unlabeled dataset loader
- SimCLR model with NT-Xent loss
- Pretraining script
- Fine-tuning script
- SLURM job scripts

**Workflow**:
1. Pretrain encoder on 100% unlabeled MSI images (200 epochs, 4-8 hours)
2. Fine-tune on 5% L2 labeled data (100 epochs, 2-4 hours)
3. Compare with baseline and hierarchical experiments

**Expected Result**: SimCLR pretraining should improve performance on 5% labeled data, potentially matching or exceeding L1→L2 hierarchical pretraining.

**Why this matters**: Demonstrates that self-supervised learning can match supervised pretraining, even without access to coarse labels!

---

**Ready to implement!** All technical details documented. No existing code changes needed. Let's build it! 🚀
