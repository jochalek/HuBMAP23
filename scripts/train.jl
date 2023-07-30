using DrWatson
@quickactivate "HuBMAP23"

using JSON3
using Images, FileIO, PolygonOps
using Metalhead, Flux, ProgressMeter, Statistics, CUDA
using DataAugmentation
using MLUtils
using JLD2

include(srcdir("utils.jl"))

# Load the jsonl
json_string = JSON3.read(datadir("exp_raw", "polygons.jsonl"); jsonlines=true);

path = datadir("exp_raw", "train")

"""
Takes a json string for a single image, loads the image and generates its mask.
"""
function grab_image_mask_pair(x)
    filepath = joinpath(path, x["id"] * ".tif")
    image = load(filepath)
    mask = build_mask(x["annotations"])
    mask = reshape(mask, size(mask)[1], size(mask)[2], 1)
    # TODO add transformations
    # tfm =
    #     ScaleFixed((256, 256)) |>
    #     PinOrigin()
    image = Image(image)
    tfms = ImageToTensor() |> Normalize((0.1, -0.2, -0.1), (1, 1, 1.0))
    timage = apply(tfms, image)
    return itemdata(timage), mask
end

_batchloader = mapobs(json_string; batched=:never) do json_string
    grab_image_mask_pair(json_string)
end;

minibatch = 8

# loader = MLUtils.DataLoader(_batchloader; batchsize=minibatch, parallel=true, collate=true) |> gpu;

loader = MLUtils.DataLoader(_batchloader; batchsize=minibatch, parallel=true, collate=true);
loader = CuIterator(loader);

# Define our model
model = UNet((512, 512), 3, 1, Metalhead.backbone(ResNet(18; pretrain=true)));

# FIXME file_exists not a real thing
# if file_exists("model.jld2")
#     model_state = JLD2.load("mymodel.jld2", "model_state")
#     Flux.loadmodel!(model, model_state)
# end

model = model |> gpu;

optim = Flux.setup(Flux.Adam(0.01), model);  # will store optimiser momentum, etc.

losses = []
@showprogress for epoch in 1:100
    for (x, y) in loader
        loss, grads = Flux.withgradient(model) do m
            # Evaluate model and loss inside gradient context:
            y_hat = m(x)
            Flux.binarycrossentropy(y_hat, y)
        end
        Flux.update!(optim, model, grads[1])
        push!(losses, loss)  # logging, outside gradient context
    end
end

println(losses)

model_state = Flux.state(model |> cpu);


jldsave("mymodel2.jld2"; model_state)

function grab_image(x::String)
    filepath = datadir("exp_raw", "test", x)
    image = load(filepath)
    image = Image(image)
    tfms = ImageToTensor() |> Normalize((0.1, -0.2, -0.1), (1, 1, 1.0))
    timage = apply(tfms, image)
    return reshape(itemdata(timage), 512, 512, 3, 1)
end

# out2 = model(noisy |> gpu) |> cpu  # first row is prob. of true, second row p(false)
