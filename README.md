# Weak label in CT image segamentation
Train models using using weak supervision such as image-level labels, producing pixel-level labels for CT scans.

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
![image](https://github.com/sino30535/Rad_AI_technical_test/blob/master/result/012_368_5_r.png)<img src="https://github.com/sino30535/Rad_AI_technical_test/blob/master/result/012_368_5.png" height="300">

![image](https://github.com/sino30535/Rad_AI_technical_test/blob/master/result/012_380_0_r.png)<img src="https://github.com/sino30535/Rad_AI_technical_test/blob/master/result/012_380_0.png" height="300">

![image](https://github.com/sino30535/Rad_AI_technical_test/blob/master/result/012_383_5_r.png)<img src="https://github.com/sino30535/Rad_AI_technical_test/blob/master/result/012_383_5.png" height="300">


# Reference
Learning Deep Features for Discriminative Localization
http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf

Grad-CAM:
Visual Explanations from Deep Networks via Gradient-based Localization
https://arxiv.org/pdf/1610.02391.pdf
