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