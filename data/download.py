from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="Zihan1004/FNSPID",
    filename="Stock_news/nasdaq_exteral_data.csv",
    repo_type="dataset",
    local_dir="./data/fnspid"
)