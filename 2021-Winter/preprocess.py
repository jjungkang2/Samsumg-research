import pandas as pd
import copy
import json
import pickle
from collections import Counter
from sklearn import preprocessing


def _ip_to_int(ip):
    octets = [int(octet) for octet in ip.split(".")]
    return (256**3)*octets[0] + (256**2)*octets[1] + (256**1)*octets[2] + (256**0)*octets[3]

    
def clean_df(df):
    df.drop(columns=df.columns[df.nunique() == 1], inplace=True) # Value 값이 모두 동일한 경우
    df.drop(columns=["Flow.ID", "Timestamp"], inplace=True) # 다른 항으로 조합할 수 있거나, 의미 없는 데이터인 경우
    df.drop_duplicates(inplace=True, ignore_index=True)

    df["Source.IP"] = df["Source.IP"].apply(_ip_to_int)
    df["Destination.IP"] = df["Destination.IP"].apply(_ip_to_int)
    df["Total.Length.of.Bwd.Packets"] = df["Total.Length.of.Bwd.Packets"].astype(int)

    return df

def filter_df(df):
    classes_to_keep = set(df["ProtocolName"].value_counts(ascending=False).head(50).index)
    protocol_encoder = {protocol: i for i, protocol in enumerate(sorted(classes_to_keep))}
    with open("dataset/protocol_name_encoding.json", "w") as f:
        json.dump(protocol_encoder, f, indent=4)

    df = df[df["ProtocolName"].isin(classes_to_keep)]
    df.reset_index(inplace=True, drop=True)
    df.loc["ProtocolName", :] = df["ProtocolName"].map(protocol_encoder)

    return df

def normalize_split_df(df):
    Y = df["ProtocolName"].values
    X = df.drop(columns=["ProtocolName"]).values

    std_scaler = preprocessing.StandardScaler()
    X_scaled = std_scaler.fit_transform(X)

    return X_scaled, Y

if __name__=='__main__':
    df = pd.read_csv('dataset/Dataset-Unicauca-Version2-87Atts.csv')
    print("===== original =====")
    print("shape:", df.shape)

    df = clean_df(df)
    print("===== after clean_df =====")
    print("shape:", df.shape)

    df = filter_df(df)
    print("===== after filter_df =====")
    print("shape:", df.shape)

    X, Y = normalize_split_df(df)
    print("===== after normalize_split_df =====")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    with open('dataset/preprocessed_dataset.pickle', 'wb') as f:
        pickle.dump(X, f)
        pickle.dump(Y, f)