# Currently not used

def get_meta_columns_dict(data_dict, meta) -> tuple:

    filtered_features = {}
    filtered_meta = {}

    def spilt_meta_features(df, meta):
        meta = [c for c in df.columns if c in meta]
        features = [c for c in df.columns if c not in meta]
        return meta, features

    for name, data in data_dict.items():
        filtered_meta[name], filtered_features[name] = spilt_meta_features(data, meta)

    return (filtered_meta, filtered_features)
