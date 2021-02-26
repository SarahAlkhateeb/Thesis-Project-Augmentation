.PHONY: dcgan dcgan_diff wgan wgan_diff dcgan_test dcgan_diff_test wgan_test wgan_diff_test  clean 


dcgan:
	bash /opt/local/bin/run_py_job.sh -e dit245_group15 -p cpu -c 4 -s dcgan.py -- --name dcgan --img_size 416

dcgan_diff:
	bash /opt/local/bin/run_py_job.sh -e dit245_group15 -p cpu -c 4 -s dcgan.py -- --name dcgan+diff --img_size 416 --diff_augment True

wgan:
	bash /opt/local/bin/run_py_job.sh -e dit245_group15 -p cpu -c 4 -s wgan_gp.py  -- --name wgan_gp --img_size 416

wgan_diff:
	bash /opt/local/bin/run_py_job.sh -e dit245_group15 -p cpu -c 4 -s wgan_gp.py  -- --name wgan_gp+diff --img_size 416

dcgan_test:
	bash /opt/local/bin/run_py_job.sh -e dit245_group15 -p cpu -c 4 -s dcgan.py -- --name dcgan --test True --num_output 1000

dcgan_diff_test:
	bash /opt/local/bin/run_py_job.sh -e dit245_group15 -p cpu -c 4 -s dcgan.py -- --name dcgan+diff --test True --num_output 1000

wgan_test:
	bash /opt/local/bin/run_py_job.sh -e dit245_group15 -p cpu -c 4 -s wgan_gp.py  -- --name wgan_gp --test True --num_output 1000

wgan_diff_test:
	bash /opt/local/bin/run_py_job.sh -e dit245_group15 -p cpu -c 4 -s wgan_gp.py  -- --name wgan_gp+diff --test True --num_output 1000

clean:
	rm slurm-*
