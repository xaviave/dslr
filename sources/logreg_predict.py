from Tools.LogReg import LogReg


def run():
    log_predict = LogReg(train=False)
    log_predict.load_theta()
    prediction = log_predict.predict(log_predict.np_df)
    log_predict.write_to_csv("houses.csv", prediction, ["Hogwarts House"])


if "__main__" == __name__:
    run()
