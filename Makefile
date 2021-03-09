.PHONY: dcgan dcgan_diff wgan wgan_diff dcgan_test dcgan_diff_test wgan_test wgan_diff_test  clean 


dcgan:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p gpu -c 4 -s dcgan.py -- --name dcgan_real 

dcgan_diff:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p gpu_shannon -c 4 -s dcgan.py -- --name dcgan+diff_real  --diff_augment True

wgan:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p gpu -c 4 -s wgan_gp.py  -- --name wgan_gp_real

wgan_diff:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p gpu_shannon -c 4 -s wgan_gp.py  -- --name wgan_gp+diff_real --diff_augment True

dcgan_test:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p cpu -c 4 -s dcgan.py -- --name dcgan_real --test True --num_output 1000

dcgan_diff_test:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p cpu -c 4 -s dcgan.py -- --name dcgan+diff_real --test True --num_output 1000

wgan_test:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p cpu -c 4 -s wgan_gp.py  -- --name wgan_gp_real --test True --num_output 1000

wgan_diff_test:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p cpu -c 4 -s wgan_gp.py  -- --name wgan_gp+diff_real --test True --num_output 1000

clean:
	rm slurm-*
