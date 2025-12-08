{
    "model_id_list": [

      "vit_base_patch8_224"
      
      "vit_base_patch16_224.augreg2_in21k_ft_in1k", # with finetuned head
      "vit_base_patch16_224", # backbone only
      
      "vit_large_patch16_rope_224",
      "vit_huge_patch16_224",
      
      "vit_large_patch14_224", # backbone only
      "vit_huge_patch14_224", # backbone only

      # DeiT3 ViT Models -img resolution 224 (Have finetuned head)
      "deit3_small_patch16_224.fb_in22k_ft_in1k",
      "deit3_base_patch16_224.fb_in22k_ft_in1k",
      "deit3_medium_patch16_224.fb_in22k_ft_in1k",
      "deit3_large_patch16_224.fb_in22k_ft_in1k",
  
      # DeiT3 ViT Models -img resolution 384
      "deit3_small_patch16_384.fb_in22k_ft_in1k",
      "deit3_base_patch16_384.fb_in22k_ft_in1k",
      "deit3_large_patch16_384.fb_in22k_ft_in1k",

      # DinoV2 
      "vit_small_patch14_dinov2",
      "vit_base_patch14_dinov2" # I finetuned this head; model checkpoint - vit_base_patch14_dinov2_in1k_best


      # DinoV2 ViT Models -img resolution 518 with registers
      "vit_small_patch14_reg4_dinov2",
      
      "vit_base_patch14_reg4_dinov2", # I finetuned this head

      # DinoV3
      "vit_base_patch16_dinov3_in1k",
      "vit_base_patch16_dinov3",# I finetuned this head # TODO: what is the difference between this and the one above?
      "vit_base_patch16_dinov3_qkvb", # 

  ""


      
    ]
  }
  