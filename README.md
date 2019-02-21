# Rad_AI_technical_test
RadAI coding challenge. Train models using using weak supervision such as image-level labels, producing pixel-level labels for CT scans.

# Dataset
Folder hierarchy of dataset.
```
--Rad_AI_technical_test
  --train.py
  --predict.py
  --train256 
    --liver 
    --noliver 
  --test256 
    --images_pngs 
    --masks_pngs 
 ```
# Perform transfer learning on CT scans dataset
```
python train.py --image_dir ./train256/
                --output_checkpoint ./output_model/checkpoint.ckpt
                --batch_size 16
                --epochs 10
```

# Generate prediction for test256/images_pngs
```
python predict.py --image_dir ./test256/images_pngs/
                  --checkpoint_path ./output_model/checkpoint.ckpt
                  --output_dir ./output_prediction
                  --sensitivity 130
```

# Result visualization
see visualization.ipynb
