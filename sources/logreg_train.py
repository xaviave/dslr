from Tools.LogReg import LogReg


def run():
    log_train = LogReg(train=True, n_iteration=30000)
    log_train.train(log_train.np_df, log_train.np_df_train)
    log_train.save_theta()


if "__main__" == __name__:
    run()
