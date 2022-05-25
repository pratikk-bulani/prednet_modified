# prednet_modified

Official Repository: https://github.com/coxlab/prednet

Problems: All the video files are merged in one .hkl file which leads to memory constraint when training on multiple videos. Rather I have changed the code such that the training videos are not merged together. Do check out the modifications done by me.

My code has adapted from: https://github.com/leido/pytorch-prednet

# Execution steps

The first step is to preprocess the data: The below code generates all the .hkl files (Each .hkl file for a video)

```
python data_preprocessing.py --videos_path <path where all the videos are kept> --output_path <path where the .hkl files should be stored>
```

The next step is to train the model: The below code trains the model

```
python train.py --input_path <path where all the .hkl files (preprocessed) for training are present> --output_path <path to save the weights> --log_dir <path for tensorboard info>
```

The last step is to perform the inference: The below code infers the model

 ```
 python test.py --input_path <path where all the .hkl files (preprocessed) for testing are present> --output_path <path where results should be saved> --weights_path <path of the trained model>
 ```
 
 The output for the inference step is sub-divided into two parts: origin (which saves the last frame of the seen input) and pred (which saves the next predicted frame)
 
 # Citation
 If you are using this code, please follow the citations of the original repository: https://github.com/coxlab/prednet
