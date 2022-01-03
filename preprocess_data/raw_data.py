#NEED TO USE OPENCE_CLONE CONDA ENVIRONMENT SINCE IT HAS PYARROW

from pathlib import Path
from datasets import load_dataset


def main(row):
    #root = Path("./jsons").glob("**/*.json")
    root = Path("/gpfs/alpine/med106/proj-shared/xf9/pubmed_data/jsons").glob("pubmed22n000*")
    #print('pubmed22n%03d*' % row)
    #root = Path('/gpfs/alpine/med106/proj-shared/xf9/pubmed_data/jsons').glob('pubmed22n%03d*' % row)
    files = [str(x) for x in root]

    dataset = load_dataset('json', data_files=files, split="train")
    print(dataset)

    d = dataset.filter(lambda example: example['abstract'] is not None)
    d = d.filter(lambda example: len(example['abstract'].split()) > 90)
    print(d)

    #d.to_json("pm001.json")
    #d.to_json('pm%03d.json' % row)
    d.to_json("pm000.json")

if __name__=="__main__":
    #for i in range(50,112):
        #main(i)
    main(0)

