# Self-training-and-Label-Propagation-for-Semi-supervised-Classification

##  training
跑對應的train_{dataset}.py來訓練不同的資料集，資料集位置預設為此repo中dataset資料集內，可在terminal內輸入`--dataset_dir`指定
其他需要哪些參數請去[`opt.py`](https://github.com/ricky-696/Self-training-and-Label-Propagation-for-Semi-supervised-Classification/blob/main/opt.py)查看

ex:
```
python train_ISIC.py --dataset_dir datasets/ISIC2018 
```

## Dataset
目前只支援MNIST, ISIC, MURA，需要train自己的dataset時，getitem需要回傳的資訊如下：
```python
    def __getitem__(self, index):
        img = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        batch = {}
        batch['img'] = img
        batch['label'] = torch.tensor(self.target[index])
        batch['omega'] = self.omega[index]
        batch['idx'] = torch.tensor(index)

        return batch
```
`img`為影像, `label`為分類的正確答案, 
`omega`為每張照片對應的權重，要在init事先宣告:
```python
self.omega = torch.tensor([1.] * len(self.data), dtype=torch.float32)
```
`idx`為__getitem__取出的index

## trainer
 - 程式主要的pipeline都[寫在這邊](https://github.com/ricky-696/Self-training-and-Label-Propagation-for-Semi-supervised-Classification/blob/b95c8b8072567ca78af74312508d35e3a14d9853/trainer.py#L616)，詳情請翻論文來看，基本上就是會產生兩個pseudo label，一個是模型產生，一個是label propagetion產生
 - 兩個pseudo label會進入[pred_pseudo_label](https://github.com/ricky-696/Self-training-and-Label-Propagation-for-Semi-supervised-Classification/blob/b95c8b8072567ca78af74312508d35e3a14d9853/trainer.py#L245)來產生真正進入訓練的pseudo label，原本MNIST學長是使用[幾層FC](https://github.com/ricky-696/Self-training-and-Label-Propagation-for-Semi-supervised-Classification/blob/b95c8b8072567ca78af74312508d35e3a14d9853/train_MNIST.py#L72)，後面發現不太合理所以改成[當模型與LP預測的pseudo label都相同，才會採用pseudo label的結果進行self-training](https://github.com/ricky-696/Self-training-and-Label-Propagation-for-Semi-supervised-Classification/blob/b95c8b8072567ca78af74312508d35e3a14d9853/Model.py#L178)。

 - 計算loss時會有兩個參數參與運算，一個是omega，一個是z，公式如下:
```python
loss = (Criterion(outputs, y_train) * omega * torch.log(total_train_data / z)).mean()
```
其中`omega`代表這個pseudo label的信心程度，如果它本身的亂度夠小，信心程度就越高
```python
batch['omega'] = torch.tensor(1 - (entropy(pre_soft[i], base=num_classes) / np.log(num_classes))) # Confidence Parameter
```
`z`代表這個data對應的類別在資料集中有幾張
```python
z = torch.index_select(num_cls_data, 0, label).to(args.device)
```
Code很亂懶的整理，只有trainer簡單模組化一下。
