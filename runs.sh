# python scripts/train_coco.py --lr 0.001 --epochs 2 --lr_decay_steps '(1, )' --model_c 50 --log_path '../exps/test/Madry adversarial R50 w Sigma=3 lr 001' --model_name 'poseresnet' --pretrained_weight_path './downloads/madry_adversarial_resnet50.pt'
# python scripts/train_coco.py --lr 0.01 --epochs 2 --lr_decay_steps '(1, )' --model_c 50 --log_path '../exps/test/Madry adversarial R50 w Sigma=3 lr 01' --model_name 'poseresnet' --pretrained_weight_path './downloads/madry_adversarial_resnet50.pt'
# python scripts/train_coco.py --lr 0.0005 --epochs 2 --lr_decay_steps '(1, )' --model_c 50 --log_path '../exps/test/Madry adversarial R50 w Sigma=3 lr 0005' --model_name 'poseresnet' --pretrained_weight_path './downloads/madry_adversarial_resnet50.pt'
# python scripts/train_coco.py --lr 0.005 --epochs 2 --lr_decay_steps '(1, )' --model_c 50 --log_path '../exps/test/Madry adversarial R50 w Sigma=3 lr 0.05' --model_name 'poseresnet' --pretrained_weight_path './downloads/madry_adversarial_resnet50.pt'

# python scripts/train_coco.py --batch_size 16 --weight_decay 0.0 --num_workers 4 --model_c 50 --image_resolution '(256, 192)' --log_path '../exps/UsingMyCustomDataset_AndOtherSetting_Madry_adversarial_eps8_R50_Sigma2' --model_name 'poseresnet' --pretrained_weight_path './downloads/madry_adversarial_resnet50.pt'
# python scripts/train_coco.py --batch_size 16 --weight_decay 0.0 --num_workers 4 --model_c 50 --image_resolution '(256, 192)' --log_path '../exps/UsingMyCustomDataset_AndOtherSetting_ST_R50_Sigma2' --model_name 'poseresnet' --pretrained_weight_path './downloads/standard_resnet50.pth'

# Benchmarking PGD acc

python scripts/test_coco_robustness.py --batch_size 16 --model_c 50 --model_name poseresnet --pretrained_weight_path '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/exps/UsingMyCustomDataset_AndOtherSetting_Madry_adversarial_eps8_R50_Sigma2/20251029_1721/checkpoint_best_mAP.pth'
python scripts/test_coco_robustness.py --batch_size 16 --model_c 50 --model_name poseresnet --pretrained_weight_path '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/exps/UsingMyCustomDataset_AndOtherSetting_ST_R50_Sigma2/20251031_1656/checkpoint_best_mAP.pth'

