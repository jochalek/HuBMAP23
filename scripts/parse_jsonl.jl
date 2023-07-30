using DrWatson
@quickactivate "HuBMAP23"

using JSON3

# Loads the jsonl
json_string = JSON3.read(datadir("exp_raw", "polygons.jsonl"); jsonlines=true);

# An example of how to index into the coordinates of the first annotation of the first image.
json_string[1][:annotations][1][:coordinates][1]

# TODO write a function that generates a mask from the jsonl file for each image
