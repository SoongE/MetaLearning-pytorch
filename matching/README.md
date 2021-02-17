# Matching Networks for One Shot Learning using Pytorch
A implementation of the NIPS 2016 paper : [Matching Networks for One Shot Learning](https://arxiv.org/pdf/1606.04080.pdf) using pytorch. In the model, somethings, such as learning rates or regression, may differ from the original paper.

I posted the details of the code in ***Korean*** on my [blog]()(will be soon.), so if you are interested, please visit!

한글로 논문과 코드에 대해 작성한 글이 있으니 관심있으신 분은 확인해보세요!

### 🚀How to run

1. #### Go into prototypical directory

   ```bash
   cd matching
   ```

2. #### Train

   ```bash
python train.py -d omniglot -m protonet
   ```
   
3. #### Test

   ```bash
python eval.py
   ```
   
4. #### Logs

   ```bash
tensorboard --logdir=runs
   ```
   

All parameters are present in `arguments.py`. If you want to adjust the parameters, modify them and run the code.

### 📈Result

| Model                | Reference Paper | This Repo |
| -------------------- | --------------- | --------- |
| Omniglot 5-w 1-s     |                 |           |
| Omniglot 5-w 5-s     |                 |           |
| Omniglot 20-w 1-s    |                 |           |
| Omniglot 20-w 1-s    |                 |           |
| miniImagenet 5-w 1-s |                 |           |
| miniImagenet 5-w 5-s |                 |           |

**Graph**

<p align="center">
    <img src="asset\omniglot_result.png" height=320>
    <img src="asset\mini_result.png" height=320>
</p>



### 📌Reference

* [Matching Networks for One Shot Learning](https://arxiv.org/pdf/1606.04080.pdf)


