# NTOU_CSE_Graduation_Project

## Dataset
1. 尾部沒有填充0對齊禎數
  - Summary_2st.xlsx
    > 修正1st的問題，亂預測的填0，整個手部沒有偵測到點的禎移除
    > 正德影片長度擷取到的特徵無法被134整除
  - Summary_3st.xlsx
    > 完全沒有正德的影片的2st更新版
  - Summary_4st.xlsx
    > 把正德有問題的影片去掉的2st更新版

2. 尾部有填充0對齊禎數
  - Summary_stuff_zero_1st.xlsx
    > 預測point的分數不夠高，亂預測的點沒有填0，整個手部都沒有偵測到點的禎數沒有移掉
  - Summary_stuff_zero_2st.xlsx
  - Summary_stuff_zero_3st.xlsx
  - Summary_stuff_zero_4st.xlsx

## Code files
- edit_excel.py
  > input: 沒有在尾部填充0的xlsx檔案
  > output: 在尾部填充0的xlsx檔案
  > > 備註: input跟output的檔案名稱要自己設定

- getMin_VideoFrame.py
  > input: 影片資料夾
  > output: 該資料夾的所有影片中最少的禎數
  > > 備註: 原本是openpose要算"幾張取一張"，但後面沒有用

- GUI.py

- is_correct_data_row.py
  > input: 沒有在尾部填充0的xlsx檔案
  > output: print出「特徵無法被134整除的是哪幾個row」

- myself_video.py
  > 很久以前的檔案，應該沒用

- myself_webcame.py
  > 很久以前的檔案，應該沒用

- test.py
  > 測試用

- timeseries_classification_from_lstm.py
  > LSTM 神經網路架構 + grad-cam

- timeseries_classification_from_scratch.py
  > Convolution 神經網路架構 + grad-cam

- timeseries_transformer_classification.py
  > Transformer 神經網路架構 + grad-cam

- vectorGenerator.py
  > openpose 擷取標準化 keypoints

