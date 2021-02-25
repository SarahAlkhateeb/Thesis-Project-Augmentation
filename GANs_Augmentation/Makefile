.PHONY: dcgan wgan clean 


dcgan:
	bash /opt/local/bin/run_py_job.sh -e dit245_group15 -p cpu -c 4 -s dcgans.py

wgan:
	bash /opt/local/bin/run_py_job.sh -e dit245_group15 -p cpu -c 4 -s wgan_gp.py  

clean:
	rm slurm-*
