import pandas as pd

lstm_data = pd.read_csv("./submission4.csv")

lstm_data["전기요금(원)"] = lstm_data["전기요금(원)"] * 0.75

lstm_data.to_csv("./remade_file.csv",index=False)


lstm_data1 = pd.read_csv("./submission_optimal (1).csv.csv")
lstm_data2 = pd.read_csv("./submission_modified.csv")

lstm_data2.columns = ["id", "전기요금(원)"]

diff_rows = lstm_data1[lstm_data1["전기요금(원)"] != lstm_data2["전기요금(원)"]]


lstm_data3 = pd.read_csv("./submission_optimal (1).csv")
lstm_data3.columns = ["id", "전기요금(원)"]
lstm_data3
lstm_data3.to_csv("./data_final.csv",index=False)


lstm_data4 = pd.read_csv("./data_final2.csv")
lstm_data4
lstm_data4.columns = ["id", "전기요금(원)"]

lstm_data4["전기요금(원)"] = lstm_data4["전기요금(원)"] * 1.2
lstm_data4.to_csv("./data_final4.csv",index=False)

lstm_data6 = lstm_data4[["id","전기요금(원)"]]
lstm_data6.to_csv("./data_final3.csv",index=False)

100/75*0.6

lstm_data6["전기요금(원)"] = lstm_data6["전기요금(원)"] * 0.8 


lstm_data7 = pd.read_csv("전기요금제출용_UTC기준.csv")
lstm_data8 =lstm_data7[["id","전기요금(원)"]]
lstm_data8.to_csv("./final_test_2.csv",index=False)