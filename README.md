# How to run backend

# backend 
python flask_backend.py --port 5001



# frontend
go to frontenf folder
npm install 
npm run dev

## Model Information

| model_id | available? | backbone only? | has registers? | input_image_resolution | patch_size | finetuned_head_path |
|----------|------------|----------------|-----------------|----------------------|------------|---------------------|
| vit_base_patch8_224 | ☐ | ☐ | ☐ | 224 | 8 | - |
| vit_base_patch16_224.augreg2_in21k_ft_in1k | ☑ | ☐ | ☐ | 224 | 16 | - |
| vit_base_patch16_224 | ☐ | ☑ | ☐ | 224 | 16 | - |
| vit_large_patch16_rope_224 | ☐ | ☐ | ☐ | 224 | 16 | - |
| vit_huge_patch16_224 | ☐ | ☐ | ☐ | 224 | 16 | - |
| vit_large_patch14_224 | ☐ | ☑ | ☐ | 224 | 14 | - |
| vit_huge_patch14_224 | ☐ | ☑ | ☐ | 224 | 14 | - |
| deit3_small_patch16_224.fb_in22k_ft_in1k | ☑ | ☐ | ☐ | 224 | 16 | - |
| deit3_base_patch16_224.fb_in22k_ft_in1k | ☑ | ☑ | ☐ | 224 | 16 | - |
| deit3_medium_patch16_224.fb_in22k_ft_in1k | ☑ | ☐ | ☐ | 224 | 16 | - |
| deit3_large_patch16_224.fb_in22k_ft_in1k | ☐ | ☐ | ☐ | 224 | 16 | - |
| deit3_small_patch16_384.fb_in22k_ft_in1k | ☐ | ☐ | ☐ | 384 | 16 | - |
| deit3_base_patch16_384.fb_in22k_ft_in1k | ☐ | ☐ | ☐ | 384 | 16 | - |
| deit3_large_patch16_384.fb_in22k_ft_in1k | ☐ | ☐ | ☐ | 384 | 16 | - |
| vit_small_patch14_dinov2 | ☐ | ☐ | ☐ | 518 | 14 | - |
| vit_base_patch14_dinov2 | ☑ | ☑ | ☐ | 518 | 14 | vit_base_patch14_dinov2_in1k_best.pth |
| vit_small_patch14_reg4_dinov2 | ☐ | ☐ | ☑ | 518 | 14 | - |
| vit_base_patch14_reg4_dinov2 | ☑ | ☑ | ☑ | 518 | 14 | vit_base_patch14_reg4_dinov2_head_finetuned_best.pth |
| vit_base_patch16_dinov3_in1k | ☐ | ☐ | ☐ | 518 | 16 | - |
| vit_base_patch16_dinov3 | ☑ | ☑ | ☐ | 518 | 16 | vit_base_patch16_dinov3_in1k_best_20_epochs.pth |
| vit_base_patch16_dinov3_qkvb | ☐ | ☐ | ☐ | 518 | 16 | - |


# TODO: 
- add multiple pytorch models support 
- use timm library
- choose model from the dropdown list
- View clicked image full screen to view finer details

UI Fixes 
+ add inverse mask 
+ change the brush size 
+ add reset mask 
+ need to pasrse the pertubed image and get the explainability map
+ host on hugging face hub