from datasets import load_dataset
from datasets import DatasetBuilder
from datasets import Dataset


in_data_dir = "audio_dataset/data/"
out_data_dir = "audio_dataset/data_HF/"




def main():
    # Load in the data
    dataset = load_dataset(
        "./",
        data_dir=in_data_dir,
        cache_dir=out_data_dir,
    )
    
    # Save the dataset for later usage
    dataset.save_to_disk(out_data_dir)
    
    
    
    
    
    
if __name__ == "__main__":
    main()