using DrWatson
@quickactivate "HuBMAP23"

using Metalhead, Flux, ProgressMeter, Statistics #Images




# Generate some data for the XOR problem: vectors of length 2, as columns of a matrix:
noisy = rand(Float32, 512, 512, 3, 32)                                    # 512x512x3×32 Array{Float32, 4}
# truth = [xor(col[1]>0.5, col[2]>0.5) for col in eachcol(noisy)]   # 1000-element Vector{Bool}
truth = falses(512, 512, 32)
for i in 1:size(noisy)[4]
    truth[:, :, i] .= any(noisy[:, :, :, i] .> 0.5, dims=3)
end

# Define our model
model = UNet((512, 512), 3, 1, Metalhead.backbone(ResNet(18; pretrain=true)))
model = model |> gpu

# model = Chain(
#     Dense(2 => 3, tanh),   # activation function inside layer
#     BatchNorm(3),
#     Dense(3 => 2),
#     softmax) |> gpu        # move model to GPU, if available

# The model encapsulates parameters, randomly initialised. Its initial output is:
# out1 = model(noisy |> gpu) |> cpu                                 # 2×1000 Matrix{Float32}

# To train the model, we use batches of 64 samples, and one-hot encoding:
# FIXME outputs wrong dimension target
# target = Flux.onehotbatch(truth, [true, false])                   # 2×1000 OneHotMatrix
target = reshape(truth, 512, 512, 1, 32)
loader = Flux.DataLoader((noisy, target) |> gpu, batchsize=4, shuffle=true);
# 16-element DataLoader with first element: (2×64 Matrix{Float32}, 2×64 OneHotMatrix)

optim = Flux.setup(Flux.Adam(0.01), model)  # will store optimiser momentum, etc.

# Training loop, using the whole data set 1000 times:
losses = []
@showprogress for epoch in 1:1_000
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

optim # parameters, momenta and output have all changed
out2 = model(noisy |> gpu) |> cpu  # first row is prob. of true, second row p(false)

mean((out2[1, :] .> 0.5) .== target)  # accuracy 94% so far!
