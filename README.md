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
- getMin_VideoFrame.py
- GUI.py
- is_correct_data_row.py
- myself_video.py
- myself_webcame.py
- test.py
- timeseries_classification_from_scratch.py
- vectorGenerator.py
