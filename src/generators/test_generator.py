from tqdm import tqdm
import pickle
from src.generators.temperate_proc_generator import TemperateProcGenerator
from src.generators.desert_proc_generator import DesertProcGenerator
from src.generators.tundra_proc_generator import TundraProcGenerator
import random

def produce_dataset(generator_class_list, num_samples=100, size=64, out_path="output.pkl"):
    """
    Given a generator subclass (TemperateProcGenerator or DesertProcGenerator),
    produce `num_samples` maps and pickle them to `out_path` as a list of dicts:
      [ { "grid_labels": <2D array>, "description": None }, … ]
    """
    generators = []
    for gen_class in generator_class_list: 
        generators.append(gen_class(size=size, seed=None))

    data = []
    i    = 0
    for _ in tqdm(range(num_samples), desc="Generating maps", unit="map"): 

        gen  = generators[i % len(generators)]
        seed = random.randint(0, 1_000_000)
        gen.generate_map(seed)
        data.append({"grid_labels": gen.grid.copy(), "description": gen.title})

    with open(out_path, "wb") as f: 
        pickle.dump(data, f)
    print(f"Saved {num_samples} maps to {out_path}")

if __name__ == "__main__":

    produce_dataset([TemperateProcGenerator, DesertProcGenerator, TundraProcGenerator], num_samples=100000, size=64, out_path="data/output/maps.pkl")
