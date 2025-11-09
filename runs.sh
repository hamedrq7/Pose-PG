# python scripts/train_coco.py --batch_size 16 --weight_decay 0.0 --num_workers 4 --model_c 50 --image_resolution '(256, 192)' --log_path '../exps/UsingMyCustomDataset_AndOtherSetting_Madry_adversarial_eps8_R50_Sigma2' --model_name 'poseresnet' --pretrained_weight_path './downloads/madry_adversarial_resnet50.pt'
# python scripts/train_coco.py --batch_size 16 --weight_decay 0.0 --num_workers 4 --model_c 50 --image_resolution '(256, 192)' --log_path '../exps/UsingMyCustomDataset_AndOtherSetting_ST_R50_Sigma2' --model_name 'poseresnet' --pretrained_weight_path './downloads/standard_resnet50.pth'
# python scripts/train_coco.py --batch_size 16 --weight_decay 0.0 --num_workers 8 --model_c 50 --image_resolution '(256, 192)' --log_path ../exps/UsingMyCustomDataset_AndOtherSetting_Madry_adversarial_eps4 --model_name poseresnet --pretrained_weight_path './downloads/madry_adversarial_resnet50_linf4.pt'
# python scripts/train_coco.py --batch_size 16 --weight_decay 0.0 --num_workers 8 --model_c 50 --image_resolution '(256, 192)' --log_path '../exps/UsingMyCustomDataset_AndOtherSetting_Madry_adversarial_l2_eps0.05' --model_name poseresnet --pretrained_weight_path './downloads/resnet50_l2_eps0.05.ckpt'

# with wd it doesnt converge whyyyy? (upto epoch 10-ish)
# python scripts/train_coco.py --batch_size 16 --weight_decay 0.0001 --num_workers 8 --model_c 50 --image_resolution '(256, 192)' --log_path '../exps/StandardPoseResnet_default_setting' --model_name 'poseresnet' --pretrained_weight_path './downloads/standard_resnet50.pth'

# Benchmarking PGD acc
# python scripts/test_coco_robustness.py --batch_size 16 --model_c 50 --model_name poseresnet --pretrained_weight_path '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/exps/UsingMyCustomDataset_AndOtherSetting_Madry_adversarial_eps8_R50_Sigma2/20251029_1721/checkpoint_best_mAP.pth'
# python scripts/test_coco_robustness.py --batch_size 16 --model_c 50 --model_name poseresnet --pretrained_weight_path '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/exps/UsingMyCustomDataset_AndOtherSetting_ST_R50_Sigma2/20251031_1656/checkpoint_best_mAP.pth'

# python scripts/test_coco_robustness.py --batch_size 16 --num_workers 8 --model_c 50 --model_name poseresnet --pretrained_weight_path '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/exps/UsingMyCustomDataset_AndOtherSetting_Madry_adversarial_l2_eps0.05/20251107_0938/checkpoint_best_mAP.pth'
# python scripts/test_ap10k_zeroshot.py --num_workers 8 --log_path temp --model_c 50 --model_name poseresnet --pretrained_weight_path '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/exps/UsingMyCustomDataset_AndOtherSetting_Madry_adversarial_l2_eps0.05/20251107_0938/checkpoint_best_mAP.pth'

# Testing Adversarial Training 
# Testing params
python scripts/train_coco_adv.py --disable_rand_init --mix_st --epsilon 0.01568 --step_size 0.01568 --weight_decay 0.0 --num_steps 1 --batch_size 16 --num_workers 8 --model_c 50 --image_resolution '(256, 192)' --log_path '../exps/AdvTraining_FGSM_MIX_eps4' --model_name 'poseresnet' --pretrained_weight_path './downloads/madry_adversarial_resnet50_linf4.pt'

