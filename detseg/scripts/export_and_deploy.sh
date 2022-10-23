
python tools/export_model.py -c configs/cascade_rcnn/cascade_rcnn_r50_vd_fpn_ssld_1x_xd.yml --output_dir inference_model/ -o weights=output/cascade_rcnn_r50_vd_fpn_ssld_1x_xd/best_model.pdparams

python deploy/python/infer.py \
	--model_dir ./inference_model/cascade_rcnn_r50_vd_fpn_ssld_1x_xd 
	--image_dir dataset/xd/det/images/ \
	--device GPU \
	--batch_size 1 \
	--output_dir ./results \
	--save_results \
	--run_benchmark
