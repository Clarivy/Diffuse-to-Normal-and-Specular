python train.py \
 --gpu_ids 1 \
 --no_instance\
 --dataroot /data/new_disk/new_disk/pangbai/d2sn_data/train/specular-data \
 --label_nc 0 \
 --output_nc 1\
 --name specular-test2 \
 --tf_log \
 --random_flip \
 --fineSize 920 \
 --random_illuminant_adjust \
 --face_color_transfer \
 --face_color_path /data/new_disk/new_disk/pangbai/d2sn_data/face_color \
 --nThreads 8 \
 --validate_freq 900 \
 --niter 150  \
 --niter_decay 150 \
 --train_mode specular \
 --no_cosine_loss \
 --validate_dataroot /data/new_disk/new_disk/pangbai/d2sn_data/test/labeled