# Kaixi
## U_NET_XCEPTION
### Optimizer
For the following (bs = 32, lr = 1e-4):
- Adam and AdamW are showing similar results (Mean Intersection Over Union: 41.23%). ReduceLROnPlateau is not showing something valuable (to be tested further).
- SGD performs worse that Adam. (Mean Intersection Over Union: 37.91%)
- SGD with exp decay (gamma = 0.9, every 7 epochs) is showing similar results to Adam: Mean Intersection Over Union: 40.21%

For the following (bs = 32, lr = 1e-2):
- SGD with exp decay (gamma = 0.96, every 10 epochs) is worse than Adam: Mean Intersection Over Union: 34.96%

For the following (bs = 32, lr = 1e-3):
- SGD with exp decay (gamma = 0.96, every 10 epochs) is worse than Adam: Mean Intersection Over Union: 39.37%. Loss is higher than Adam (first point)

For the following (bs = 16, lr = 1e-4):
- AdamW (Mean Intersection Over Union: < 41.23%)

## U_NET
### Optimizer
For the following (bs = 32, lr = 1e-4):
- AdamW (Mean Intersection Over Union: 43.05%)

### Augmentation
`X_train`  has been doubled hence using `p=1.0` for all the augmentation in the pipeline
AdamW[lr = 1e-4] + sparse_cross_ent + bs[16] + aug:
```
transform = A.Compose([
      A.GridElasticDeform(num_grid_xy=(8, 8), magnitude=10), # p = 1
      A.XYMasking(),
      A.ShiftScaleRotate()

  ])
```
- Validation MIOU: 0.4654
- Competition MIOU: 0.4454

`X_train`  has been doubled hence using `p=1.0` for all the augmentation in the pipeline
AdamW[lr = 1e-4] + sparse_cross_ent + bs[16] + aug:
```
transform = A.Compose([
      A.GridElasticDeform(num_grid_xy=(4, 4), magnitude=10), # p = 1
      A.XYMasking(),
      A.ShiftScaleRotate()

  ])
```
- Validation MIOU: 0.45x
- Competition MIOU: 0.4483

## Issues
- Dice seems not be working out of the box (tf graph execution error good luck)

# Francesco

## UWNet
- Tried AdamW with default settings and aug and got Final validation Mean Intersection Over Union: 48.68%
  - Validation Cross-entropy falling after 3 epochs and then juggling around and even rising up, while training crossentropy goes down to zero steadly in over 90 epochs
  - Validation accuracy and training accuracy follow the same path as loss but in the other way
  - MIOU as accuracy
  - All suggest a lack of generalization. 
  - On test: Test Accuracy: 0.7388 Test Mean Intersection over Union: 0.4122
  - Public score on Kaggle 0.45617 > 0.44827 (best)

- Second attempt:
  - Changes:
    - Batch size = 64
    - Aggressive (but more random) albumentation (
      A.ShiftScaleRotate(p=0.7),
      A.RandomBrightnessContrast(p=0.7),  
      A.GaussianBlur((3,5), p=0.5), 
      A.GridDropout(p=0.5), 
      A.GridElasticDeform(num_grid_xy=(8, 8), magnitude=5, p=0.5),
      A.XYMasking(p=0.5)  
      )
    - 0.2 dropouts in the pre-attention blocks (1-5) after all first activations
    - Worse than the first attempt, stop early, wobbly acc

- Third:
  - Used Luca's Aug (0.463 on kaggle) but with my dropout and batch size
  - Touch 0.47 while training but still super wobbly

- Fourth
  - BS: 32
  - LR_onPlateau patience 20 factor 0.5 (half)
  - GPT proposed aug slightly modified:

    transform = A.Compose([
            A.RandomRotate90(p=0.5),  # Random 90-degree rotation
            A.HorizontalFlip(p=0.5),  # Horizontal flip for diverse texture representation
            A.VerticalFlip(p=0.5),  # Vertical flip to simulate different orientations
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Adjust brightness and contrast
            A.GaussianBlur(blur_limit=3, p=0.2),  # Add blur to simulate camera effects
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5),  # Randomly occlude parts of the image
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),  # Random shifts, scales, and rotations
            A.ElasticTransform(alpha=1, sigma=50, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=0.5),
            A.Resize(height=IMG_SIZE[0], width=IMG_SIZE[1], p=1),  # Resize for consistent input size
        ])

  - Moved dropout after the second activation in each block, upped to 0.3
  - No relevant updates

- Fifth (ASPP)
  - LR_onPlateau -> 0.8 every 25
  - Atrous Spatial Pyramid Pooling + Spatial Dropout
  - Results are in the order of the best so far but we have a much more ordinate convorgence

# Arianna

- ASPP with weights initialization with He normal + l2 regularization lambda=1e-4 + lr reduce of factor 0.5 with patience 20: Final validation Mean Intersection Over Union: 48.46%. Test Mean Intersection over Union: 0.4762. On kaggle = 0.47093

- RockSeg (same settings): Final validation Mean Intersection Over Union: 49.05%. Test Mean Intersection over Union: 0.4106. On kaggle: 0.40391.
  Problem: validation goes up and down a lot