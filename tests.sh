# python scripts/test_model_weights.py --exp_name '../exps/STPoseResnet/weight_ananlysis' --pretrained_weight_path 'D:/Pose/exps/STPoseResnet/checkpoint_best_mAP.pth' --model_c 50 --model_nof_joints 17 --model_name poseresnet
# python scripts/test_model_weights.py --exp_name '../exps/AdvPoseResnet/weight_ananlysis' --pretrained_weight_path 'D:/Pose/exps/AdvPoseResnet/checkpoint_best_mAP.pth' --model_c 50 --model_nof_joints 17 --model_name poseresnet
# python scripts/test_model_weights.py --exp_name '../exps/FastFGSMTraining/weight_ananlysis' --pretrained_weight_path 'D:/Pose/exps/FastFGSMTraining/checkpoint_best_mAP.pth' --model_c 50 --model_nof_joints 17 --model_name poseresnet
# python scripts/test_model_weights.py --exp_name '../exps/ViTPose_small_downloaded/weight_ananlysis' --pretrained_weight_path 'D:/Pose/exps/ViTPose_small_downloaded/vitpose_small.pth' --model_name vitpose_small

# python scripts/test_coco.py --exp_name 'testing_vit' --num_workers 2 --pretrained_weight_path '' --model_name 'vitpose_small'


# python scripts/test_crowdpose_zeroshot.py --log_path '../exps/vitpose_small' --pretrained_weight_path './downloads/vitpose_small.pth' --model_name 'vitpose_small' --exp_name 'testing_crowdpose_zeroshot' --num_workers 8 
python scripts/test_crowdpose_zeroshot.py --model_c 50 --model_nof_joints 17 --log_path '../exps/st_poseresnet' --pretrained_weight_path '../exps/UsingMyCustomDataset_AndOtherSetting_ST_R50_Sigma2/20251031_1656/checkpoint_best_mAP.pth' --model_name 'poseresnet' --exp_name 'testing_crowdpose_zeroshot' --num_workers 8 
python scripts/test_crowdpose_zeroshot.py --model_c 50 --model_nof_joints 17 --log_path '../exps/fgsmtraining_poseresnet' --pretrained_weight_path '../exps/AdvTraining_FAST_FGSM_RS_MIX_eps4_stepsize_2/20251109_1807/checkpoint_cln_best_mAP.pth' --model_name 'poseresnet' --exp_name 'testing_crowdpose_zeroshot' --num_workers 8 

