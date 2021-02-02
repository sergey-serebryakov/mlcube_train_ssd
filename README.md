# MLCommons Training SSD MLCube

> WORK IN PROGRESS

> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
> 
> This project contains files from MLCommons [Training Repository](https://github.com/mlcommons/training), in
> particular, the following directory:
> 
> https://github.com/mlcommons/training/tree/master/single_stage_detector
> 
> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Clone this project
```shell
git clone https://github.com/sergey-serebryakov/mlcube_train_ssd
cd ./mlcube_train_ssd
```

Create python virtual environment
```shell
virtualenv -p python3 ./env
source ./env/bin/activate
pip install mlcube mlcube-docker
```

Patch MLCube docker runner. Edit the following file:
```shell
./env/lib/python3.8/site-packages/mlcube_docker/docker_run.py
```
Replace line [#70](https://github.com/mlcommons/mlcube/blob/master/runners/mlcube_docker/mlcube_docker/docker_run.py#L70) (`cmd = f"docker run ..."`) with the following line:
```shell
cmd = f"docker run --rm {runtime_arg} -v /dev/shm:/dev/shm --net=host --privileged=true {volumes_str} {env_args} {image_name} {args}"
```
PyTorch data loaders need more shared memory, and it's work in progress to support docker parameters.

Configure MLCube:
```shell
mlcube_docker configure --mlcube=. --platform=platforms/docker.yaml
```
If you already have coco2017 dataset, copy it to `./workspace/data`. This folder should contain three sub-folders:
`train2017`, `val2017` and `annotations`. If you do not have the dataset, this MLCube can download it. You will need
a little over 40 GB. The downloaded files will be stored in `./workspace/cache`, and the dataset in `./workspace/data`
(I have not really tested the download task):
```shell
mlcube_docker run --mlcube=. --platform=platforms/docker.yaml --task=run/download.yaml
```

Run benchmark for one epoch:
```shell
mlcube_docker run --mlcube=. --platform=platforms/docker.yaml --task=run/train.yaml
```

Currently, it runs with the following error (to be solved):
```shell
Iteration:      0, Loss function: 22.812, Average Loss: 0.023
Iteration:    100, Loss function: 9.449, Average Loss: 1.472
Iteration:    200, Loss function: 8.723, Average Loss: 2.195
Iteration:    300, Loss function: 8.467, Average Loss: 2.814
Iteration:    400, Loss function: 8.482, Average Loss: 3.351
Iteration:    500, Loss function: 8.038, Average Loss: 3.822
Traceback (most recent call last):
  File "train.py", line 431, in <module>
    main()
  File "train.py", line 424, in main
    success = train300_mlperf_coco(args)
  File "train.py", line 340, in train300_mlperf_coco
    for nbatch, (img, img_id, img_size, bbox, label) in enumerate(train_dataloader):
  File "/opt/conda/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 363, in __next__
    data = self._next_data()
  File "/opt/conda/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 971, in _next_data
    return self._process_data(data)
  File "/opt/conda/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 1014, in _process_data
    data.reraise()
  File "/opt/conda/lib/python3.7/site-packages/torch/_utils.py", line 395, in reraise
    raise self.exc_type(msg)
FileNotFoundError: Caught FileNotFoundError in DataLoader worker process 2.
Original Traceback (most recent call last):
  File "/opt/conda/lib/python3.7/site-packages/torch/utils/data/_utils/worker.py", line 185, in _worker_loop
    data = fetcher.fetch(index)
  File "/opt/conda/lib/python3.7/site-packages/torch/utils/data/_utils/fetch.py", line 44, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/opt/conda/lib/python3.7/site-packages/torch/utils/data/_utils/fetch.py", line 44, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/mlperf/ssd/utils.py", line 585, in __getitem__
    img = Image.open(img_path).convert("RGB")
  File "/opt/conda/lib/python3.7/site-packages/PIL/Image.py", line 2580, in open
    fp = builtins.open(filename, "rb")
FileNotFoundError: [Errno 2] No such file or directory: '/mlcube_io0/data/train2017/000000182825.jpg'

```