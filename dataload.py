import pandas as pd
from datasets import Dataset

def mytok(seq, kmer_len, s):
    seq = seq.upper().replace("T", "U")    
    kmer_list = []
    for j in range(0, (len(seq)-kmer_len)+1, s):
        kmer_list.append(seq[j:j+kmer_len])
    return kmer_list

########### loading dp dataset
def build_dp_dataset():
    def load_dataset(data_path, split):
        df = pd.read_csv(data_path)
        df = df[df["split"] == split]
        df = df.dropna(subset=["bp_zscore"])
        
        # df['utr5_size'] = df['UTR5'].astype(str).map(len)
        # df['cds_size'] = df['CDS'].astype(str).map(len)
        # df['utr3_size'] = df['UTR3'].astype(str).map(len)
        # df = df[df['utr5_size'] <= 512]
        # df = df[df['cds_size'] <= 1020]
        # df = df[df['utr3_size'] <= 1024]
            
        utr5 = df["UTR5"].values.tolist()
        utr3 = df["UTR3"].values.tolist()
        cds = df["CDS"].values.tolist()
        ys = df["bp_zscore"].values.tolist()
        
        utr5 = [" ".join(mytok(seq, 1, 1)) for seq in utr5]
        cds  = [" ".join(mytok(seq, 3, 3)) for seq in cds]
        utr3 = [" ".join(mytok(seq, 1, 1)) for seq in utr3]
        seqs = list(zip(utr5, cds, utr3))
        
        assert len(seqs) == len(ys)
        
        return seqs, ys
    
    train_seqs, train_ys = load_dataset("data/translation_rate.csv", "train")
    valid_seqs, valid_ys = load_dataset("data/translation_rate.csv", "valid")
    test_seqs, test_ys = load_dataset("data/translation_rate.csv", "test")

    ds_train = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(train_seqs, train_ys)])
    ds_valid = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(valid_seqs, valid_ys)])
    ds_test  = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(test_seqs, test_ys)])

    return ds_train, ds_valid, ds_test

def build_class_dataset():
    def load_dataset(data_path, split_num):  # Changed from split to split_num
        df = pd.read_csv(data_path)
        df = df[df["split"] == split_num]  # Filter by numeric split
        df = df.fillna('')  # Handle NaN values like in the working function
        df = df.dropna(subset=["ClassificationID"])  # Drop rows with missing labels
        
        utr5 = df["5' UTR"].values.tolist()
        utr3 = df["3' UTR"].values.tolist()
        cds = df["CDS"].values.tolist()
        ys = df["ClassificationID"].values.tolist()
        
        utr5 = [" ".join(mytok(seq, 1, 1)) for seq in utr5]
        cds = [" ".join(mytok(seq, 3, 3)) for seq in cds]
        utr3 = [" ".join(mytok(seq, 1, 1)) for seq in utr3]
        
        seqs = list(zip(utr5, cds, utr3))
        assert len(seqs) == len(ys)
        return seqs, ys
    
    # Map your available splits (1, 2, 3, 4, 5) to train/valid/test
    # Can adjust these based on your preferred split strategy
    train_splits = [1, 2, 3]  # Use splits 1, 2, 3 for training
    valid_split = 4           # Use split 4 for validation
    test_split = 5            # Use split 5 for testing
    
    # Load and combine training data from multiple splits
    train_seqs, train_ys = [], []
    for split_num in train_splits:
        seqs, ys = load_dataset("data/protein_expression_5class.csv", split_num)
        train_seqs.extend(seqs)
        train_ys.extend(ys)
    
    valid_seqs, valid_ys = load_dataset("data/protein_expression_5class.csv", valid_split)
    test_seqs, test_ys = load_dataset("data/protein_expression_5class.csv", test_split)
    
    ds_train = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(train_seqs, train_ys)])
    ds_valid = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(valid_seqs, valid_ys)])
    ds_test = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(test_seqs, test_ys)])
    
    return ds_train, ds_valid, ds_test

def build_liver_dataset():
    def load_dataset(data_path, split):
        df = pd.read_csv(data_path)
        df = df[df["split"] == split]

        utr5 = df["5' UTR"].values.tolist()
        utr3 = df["3' UTR"].values.tolist()
        cds = df["CDS"].values.tolist()
        ys = df["Liver_norm"].values.tolist()
        
        utr5 = [" ".join(mytok(seq, 1, 1)) for seq in utr5]
        cds  = [" ".join(mytok(seq, 3, 3)) for seq in cds]
        utr3 = [" ".join(mytok(seq, 1, 1)) for seq in utr3]
        seqs = list(zip(utr5, cds, utr3))
        
        assert len(seqs) == len(ys)
        
        return seqs, ys

    train_seqs, train_ys = load_dataset("data/transcript_expression_liver.csv", "train")
    valid_seqs, valid_ys = load_dataset("data/transcript_expression_liver.csv", "valid")
    test_seqs, test_ys = load_dataset("data/transcript_expression_liver.csv", "test")

    ds_train = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(train_seqs, train_ys)])
    ds_valid = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(valid_seqs, valid_ys)])
    ds_test  = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(test_seqs, test_ys)])

    return ds_train, ds_valid, ds_test

def build_saluki_dataset(cross):
    def load_dataset(data_path, split_num):  # Changed from split to split_num
        df = pd.read_csv(data_path)
        df = df[df["split"] == split_num]  # Filter by numeric split
        df = df.fillna('')
        df = df.dropna(subset=["y"])
            
        utr5 = df["UTR5"].values.tolist()
        utr3 = df["UTR3"].values.tolist()
        cds = df["CDS"].values.tolist()
        ys = df["y"].values.tolist()
        
        utr5 = [" ".join(mytok(seq, 1, 1)) for seq in utr5]
        cds  = [" ".join(mytok(seq, 3, 3)) for seq in cds]
        utr3 = [" ".join(mytok(seq, 1, 1)) for seq in utr3]
        seqs = list(zip(utr5, cds, utr3))
        
        assert len(seqs) == len(ys)
        
        return seqs, ys

    # Use numeric splits - can adjust these based on how you want to split
    # For example, use splits 0-7 for train, 8 for valid, 9 for test
    train_splits = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 splits for training
    valid_split = 8
    test_split = 9
    
    # Load and combine training data from multiple splits
    train_seqs, train_ys = [], []
    for split_num in train_splits:
        seqs, ys = load_dataset("data/mrna_half-life.csv", split_num)
        train_seqs.extend(seqs)
        train_ys.extend(ys)
    
    valid_seqs, valid_ys = load_dataset("data/mrna_half-life.csv", valid_split)
    test_seqs, test_ys = load_dataset("data/mrna_half-life.csv", test_split)

    ds_train = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(train_seqs, train_ys)])
    ds_valid = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(valid_seqs, valid_ys)])
    ds_test  = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(test_seqs, test_ys)])

    return ds_train, ds_valid, ds_test