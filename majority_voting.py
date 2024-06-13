import logging
import datetime
from datetime import date, timedelta
import pandas as pd



def mv(df, debug=False):
    """
    Compute the majority voting for a given prediction of a BERT model and display accuracy and F1 after applying majority voting.
    (The function does not change predictions for users with no predicted majority for one gender.)

    Parameters:
        df (pd.DataFrame): Pandas DataFrame with ['Gender', 'ReviewText'] as columns.

        debug (bool): Specifies whether debug messages should be on (True) or not (False). Default is False.

    Returns:
        Accuracy, F1 (male), F1 (female) as console output(!)
    """
    if debug:
        logging.basicConfig(format='%(asctime)s [%(levelname)s] - %(message)s', datefmt='%d-%m-%y %H:%M:%S', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(asctime)s [%(levelname)s] - %(message)s', datefmt='%d-%m-%y %H:%M:%S', level=logging.INFO)
    log_starttime = datetime.datetime.now()
    logging.info("Computing majority voting..")

    # Assuming Gender is binary (1/0), calculate majority voting
    df['majority_voting'] = df['Gender']

    logging.info("Computing majority voting stats..")
    logging.info("")
    logging.info("----------------------------------")
    acc = mv_stats_acc(df)
    logging.info("Accuracy = {:.4}".format(acc))
    f1_male = mv_stats_f1(df, 1)
    logging.info("F1 male = {:.4}".format(f1_male))
    f1_female = mv_stats_f1(df, 0)
    logging.info("F1 female = {:.4}".format(f1_female))
    logging.info("----------------------------------")
    logging.info("")
    log_endtime = datetime.datetime.now()
    log_runtime = (log_endtime - log_starttime)
    logging.info("Total runtime: " + str(log_runtime))



def mv_verify(df, ids):
    cnt_inconsistencies = 0
    for userid in ids:
        if len(list(set(df.loc[df["userid"] == userid]["label"]))) > 1:
            logging.debug("No unique label assignment for user with ID {} !".format(userid))
            cnt_inconsistencies += 1
    return cnt_inconsistencies



def mv_compute(df, ids):
    df_mv = df.copy()
    df_mv["majority_voting"] = df["prediction"]
    cnt_unchanged_users = 0
    cnt_unchanged_reviews = 0
    for userid in ids:
        predictions = list(df.loc[df["userid"] == userid]["prediction"])
        if (predictions.count(0) != predictions.count(1)) and (len(predictions) > 1):
            mv = max(predictions, key=predictions.count)
            df_mv.loc[df_mv["userid"]==userid, "majority_voting"] = mv
        else:
            cnt_unchanged_users += 1
            cnt_unchanged_reviews += len(predictions)
    logging.info("Done! Majority voting rejected for {} review(s) ({:.2%}) of {} user(s) ({:.2%}).".format(
        cnt_unchanged_reviews, 
        cnt_unchanged_reviews/df.shape[0], 
        cnt_unchanged_users, 
        cnt_unchanged_users/len(ids))
    )
    logging.debug("Overview (Prediction vs. Majority Voting):")
    if logging.DEBUG >= logging.root.level:
        print(df_mv)
    return df_mv



def mv_stats_acc(df):
    males = df[df['Gender'] == 1]
    females = df[df['Gender'] == 0]
    correct_male = (males['majority_voting'] == 1).sum()
    correct_female = (females['majority_voting'] == 0).sum()
    total_male = males.shape[0]
    total_female = females.shape[0]
    accuracy = (correct_male + correct_female) / (total_male + total_female)
    return accuracy



def mv_stats_f1(df, key_pos, pred_label="majority_voting"):
    pos = df[df['label'] == key_pos]  # Use 'label' instead of 'Gender'
    neg = df[df['label'] != key_pos]
    TP = (pos[pred_label] == key_pos).sum()  # Use pred_label for predictions
    FP = (neg[pred_label] == key_pos).sum()
    FN = (pos[pred_label] != key_pos).sum()
    return TP / (TP + 0.5 * (FP + FN))



# Example where MV can improve accuracy for user 1 but skips user 2:
# mv(pd.DataFrame(data={"userid": [1, 1, 2, 1, 2], "label": [0, 0, 1, 0, 1], "prediction": [0, 0, 1, 1, 0]}), True)
# Example where MV gets corrupted input data, as user 1 has different label assignments:
# mv(pd.DataFrame(data={"userid": [1, 1, 2, 1, 2], "label": [1, 0, 1, 0, 1], "prediction": [0, 0, 1, 1, 0]}), True)