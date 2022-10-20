import sys

abcdefg = True
sys.path.append(".")
# sys.path.append("..")
if abcdefg:
    import preprocess_userCSV


# /* Input START*/
models = [
    "Conv",
    "Transformer_best_model.h5",
]
path = "../testtest/Video.mov"
filename = path.split("/")[-1]
# /* Input END */
preprocess_userCSV.preprocess(max_column=27301, src_csv_file='webcam.csv')
