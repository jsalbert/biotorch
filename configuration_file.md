Configuration files are structured in 6 parts as shown in the following example:

```yaml
# 1. Experiment 
experiment:
  name (string): Name of the experiment.
  output_dir (string):  Folder where the outputs of the experiment will be saved.
  # For reproducibility
  seed (int): Random seed for the experiment.
  deterministic (boolean): If true, will apply PyTorch deterministic operations to guarantee reproducibility.
```

```yaml
# 2. Data
data:
  dataset (string): Name of the dataset
  dataset_path (string): If dataset is not supported by Torchvision, a path can be provided (e.g., ImageNet)
  target_size (int): Images will be resized to the target size (if specified): 32
  num_workers (int): Number of threads for data loading 
```

```yaml
# 3. Model
model:
  architecture (string): Name of the model architecture 
  mode:
    type (string): Name of the feedback alignment method 
    options:
      init (string): Layer initialization method
      gradient_clip (boolean): If true, will apply gradient clipping (-1, 1)

  pretrained (boolean): If true, will load a pre-trained model and copy the original weights to the new layers
  # checkpoint (string): Path to a custom model
  loss_function:
    name (string) : Name of the loss function
```

```yaml
# 4. Training Parameters
training:
  hyperparameters:
    epochs (int): Number of epochs to run the training
    batch_size (int): Data batch size

  optimizer:
    type: "SGD"
    lr: 0.1
    weight_decay: 0.0001
    momentum: 0.9

  lr_scheduler:
    type: "multistep_lr"
    gamma: 0.1
    milestones: [100, 150, 200]

  metrics:
    top_k: 5
    display_iterations: 500
    weight_alignment (boolean): Track the angle between layers forward and backward weights
    weight_ratio (boolean): Track the norm ratio between layers forward and backward weights
```

```yaml
# 5. Infrastructure
infrastructure:
  gpus (int/list): If -1 will use CPU, else will use the corresponding GPUs devices
```

```yaml
# 6. Evaluation
evaluation (boolean): If true, it will run an evaluation in the test and store the metrics after the training
```
