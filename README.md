# Correspondence Matters for Video Referring Expression Comprehension

PyTorch Implementation of paper:

> **Correspondence Matters for Video Referring Expression Comprehension (ACM MM 2022)**
>
> Meng Cao, Ji Jiang, Long Chen and Yuexian Zou.
>
>[[ArXiv](https://arxiv.org/abs/2207.10400)]

## Updates
* [Aug. 2022] We have released the code.
* [Jul. 2022] We will release the code asap. No later than the end of August.

## Dependencies
* Python 3
* Pytorch >= 0.4.1
* Others (OpenCV, scipy, etc.)

## Code and Data Preparation
* Download VID Dataset from this [link](<https://bvisionweb1.cs.unc.edu/ilsvrc2015/>) and put it into ./data/VID.
* Download VID Dataset split file from this [link](<https://drive.google.com/file/d/1CYxHPI04ScdWQlmcWNxjYtAiu2m041Vq/view>); Unzip it and put it into ./data/.
* The structure of ./data/ is as follows.
```
├── VID
├── VID_video_level_test.pth
└── VID_video_level_train.pth
```


## Training
* Train DCNet on VID 
```
python -m torch.distributed.launch --nproc_per_node 4  train_DCNet.py  --data_root  ./data  --dataset VID --gpu 0,1,2,3  --savename SAVE_NAME   --batch_size 8  --lstm
```

## Testing
* Test DCNet on VID :

```
python test_DCNet.py  --data_root  ./data  --dataset VID --gpu GPUID --savename DCNet  --batch_size BATCHSIZE --lstm --test --resume ./saved_models/DCNet.pth.tar
```

## Post processing (Optional)

There is a slight improvement in performance with this post-processing step.

* Generate cache files
```
python test_DCNet.py  --data_root  ./data  --dataset VID --gpu GPUID --savename DCNet  --batch_size BATCHSIZE --lstm --cache --resume ./saved_models/DCNet.pth.tar
```
* Post-process the initial results
```
python post_processing.py --data_root ./data/ --dataset VID --gpu GPUID --savename model_post_processing --batch_size 16 --lstm --test --cache_dir CACHE_FILE_PATH
```


## ToDo

```markdown
- [ ] More dataset support
- [ ] Pre-trained weight
```


## Citation
Please **[★star]** this repo and **[cite]** the following paper if you feel our CoLA useful to your research:
```
@article{cao2022correspondence,
  title={Correspondence Matters for Video Referring Expression Comprehension},
  author={Cao, Meng and Jiang, Ji and Chen, Long and Zou, Yuexian},
  journal={the 30th ACM International Conference on Multimedia},
  year={2022}
}

@article{cao2022correspondence,
  title={Correspondence Matters for Video Referring Expression Comprehension},
  author={Cao, Meng and Jiang, Ji and Chen, Long and Zou, Yuexian},
  journal={arXiv preprint arXiv:2207.10400},
  year={2022}
}
```