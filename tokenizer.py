import logging
import pandas as pd
import torch
import datetime
from transformers import BertTokenizerFast, AlbertTokenizer, AutoTokenizer
from sklearn import preprocessing

def prepare_data(
    file_data,
    return_data=True,
    max_tokencount=510,
    truncating_method="head",
    file_results=None,
    num_rows=None,
    embedding_type="bert",
    dataset_type="amazon",
):
    """
    Prepare the data for the BERT model.
    """
    
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] - %(message)s",
        datefmt="%d-%m-%y %H:%M:%S",
        level=logging.INFO,
    )
    log_starttime = datetime.datetime.now()
    
    # Load the data
    logging.info(f"Loading data from {file_data}...")
    try:
        data = pd.read_csv(file_data, nrows=num_rows)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return
    
    logging.info("Data loaded successfully.")
    logging.info(f"First few rows of the data:\n{data.head()}")
    
    # Ensure the 'Gender' column is numeric
    if "Gender" in data.columns:
        logging.info("Converting 'Gender' column to numeric...")
        data["Gender"] = pd.to_numeric(data["Gender"], errors="coerce")
        # Drop any rows where Gender conversion failed
        if data["Gender"].isnull().any():
            logging.warning("Some 'Gender' values could not be parsed. Dropping those rows.")
            data = data.dropna(subset=["Gender"])
            data["Gender"] = data["Gender"].astype(int)  # Ensure no NaNs and correct dtype after dropping
    else:
        logging.error("Gender column not found in data.")
        return
    
    # Log statistics
    logging.info("Total: {}".format(len(data)))
    logging.info("Male: {} ({:.2%})".format(len(data[data["Gender"] == 1]), len(data[data["Gender"] == 1]) / len(data)))
    logging.info("Female: {} ({:.2%})".format(len(data[data["Gender"] == 0]), len(data[data["Gender"] == 0]) / len(data)))
    
    # Load the tokenizer based on embedding type
    logging.info("Loading {} tokenizer ...".format(embedding_type))
    tokenizer_map = {
        "bert": BertTokenizerFast.from_pretrained("bert-base-uncased"),
        "albert": AlbertTokenizer.from_pretrained("albert-base-v1"),
        "sentiment_bert": BertTokenizerFast.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment"),
        "custom_bert": BertTokenizerFast.from_pretrained("bert-base-uncased"),  # Use BertTokenizerFast
    }
    
    if embedding_type not in tokenizer_map:
        logging.error("Unknown embedding type!")
        return
    
    tokenizer = tokenizer_map[embedding_type]
    
    # Truncation strategy
    max_tokencount = min(max_tokencount, 510)
    selector = {
        "head": lambda x: x[:max_tokencount + 1] + x[-1:],
        "tail": lambda x: x[:1] + x[-max_tokencount - 1:],
        "headtail": lambda x: x[:max_tokencount // 4] + x[-(max_tokencount - max_tokencount // 4):],
    }
    logging.info("Tokenizer loaded!")
    
    # Tokenize, truncate oversized data and apply padding
    logging.info("Applying tokenizer ...")
    cnt_oversized = 0
    
    def tokenize_and_truncate(text):
        encoding = tokenizer.encode_plus(text, max_length=max_tokencount, truncation=truncating_method)
        tokens = encoding["input_ids"]
        if len(tokens) > max_tokencount:
            nonlocal cnt_oversized
            cnt_oversized += 1
            tokens = selector[truncating_method](tokens)
        tokens += [0] * (max_tokencount - len(tokens) + 2)  # Padding
        return tokens
    
    data["ReviewText"] = data["ReviewText"].apply(tokenize_and_truncate)
    logging.info("Tokenization done!")
    logging.info("{} reviews ({:.2%}) were oversized and truncated".format(cnt_oversized, cnt_oversized / len(data)))
    
    # Attention mask
    def create_attention_mask(tokens):
        return [1 if token > 0 else 0 for token in tokens]
    
    data["att_mask"] = data["ReviewText"].apply(create_attention_mask)
    
    log_endtime = datetime.datetime.now()
    log_runtime = (log_endtime - log_starttime)
    logging.info("Total runtime: " + str(log_runtime))
    
    # Prepare the results for torch.save
    results = {
        "input_ids": torch.tensor(list(data["ReviewText"])),
        "attention_mask": torch.tensor(list(data["att_mask"])),
        "target": torch.tensor(data["Gender"].astype(int).tolist()),
    }
    
    # Include user IDs if present
    if "UserId" in data.columns:
        id_list = list(data["UserId"])
        le = preprocessing.LabelEncoder()
        user_ids = le.fit_transform(id_list)
        results["user_id"] = torch.tensor(user_ids)
    
    # Save to file if specified
    if file_results:
        torch.save(results, file_results)
    
    # Return results if needed
    if return_data:
        return results
