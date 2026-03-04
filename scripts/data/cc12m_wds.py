# scripts/data/cc12m_wds.py
import webdataset as wds
from PIL import Image

def build_cc12m_wds(shards_pattern: str, transform, tokenize, shuffle_buf=10000):
    # 多机/多卡安全切分：split_by_node + split_by_worker
    dataset = (
        wds.WebDataset(shards_pattern, resampled=True)
        .shuffle(shuffle_buf)
        .decode("pil")
        .to_tuple("jpg", "txt")
        .map(lambda x: (transform(x[0].convert("RGB")), x[1]))
        .map(lambda x: (x[0], tokenize([x[1]])[0]))  # tokenize to 1D token, later stack
    )
    return dataset
