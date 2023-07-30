"""
Takes a list of vertices, poly,

...
vertices = json_string[1][:annotations][1][:coordinates][1]

and fills the image, img, with the polygon.
"""
function fill_a_poly!(img, poly)
    _poly = stack(poly)
    (xmin, xmax), (ymin, ymax) = extrema(_poly, dims=2) # get the box the polygon fits in
    poly = [[i, j] for (i, j) in poly] # convert to array of arrays
    for i in xmin:xmax
        for j in ymin:ymax
            if inpolygon([i, j], poly; in=1, on=1, out=0) == 1
                img[j+1, i+1] = true
            end
        end
    end
end

"""
Takes a set of image annotation coordinates and returns a mask for all
annotations of label_type (default "blood_vessel").
"""
function build_mask(x; size=(512, 512), type=Float32, label_type="blood_vessel")
    # mask = zeros(type, size)
    mask = falses(size)
    for i in 1:length(x)
        if x[i][:type] == label_type
            fill_a_poly!(mask, x[i][:coordinates][1])
        end
    end
    return mask
end


"""
Run length encodes a mask. The output should be the same as python's pycocotools._mask.encode()
"""
function encode_mask(mask::AbstractArray)
    # ... TODO
    return rle
end



#FIXME This is temporary. I don't actually want to generate images of all mask
# I just want to use this for visually checking it
function gen_masks(df)
    #classes = Dict("prostate"=>Gray{N0f8}(0.03125), "spleen"=>Gray{N0f8}(0.0625), "lung"=>Gray{N0f8}(0.09375), "kidney"=>Gray{N0f8}(0.125), "largeintestine"=>Gray{N0f8}(0.15625))
    classes = Dict("prostate" => ColorTypes.Gray{FixedPointNumbers.N0f8}(0.004),
        "spleen" => ColorTypes.Gray{FixedPointNumbers.N0f8}(0.004),
        "lung" => ColorTypes.Gray{FixedPointNumbers.N0f8}(0.004),
        "kidney" => ColorTypes.Gray{FixedPointNumbers.N0f8}(0.004),
        "largeintestine" => ColorTypes.Gray{FixedPointNumbers.N0f8}(0.004))
    if isdir(datadir("exp_pro", "masks"))
    else
        mkdir(datadir("exp_pro", "masks"))
    end
    maskdir(args...) = projectdir(datadir("exp_pro", "masks"), args...)

    for i in 1:nrow(df)
        mask = masks_from_dataframe(df, i, classes)
        filename = df[i, :id]
        if isfile(maskdir("$filename" * ".png"))
            println("not overwriting $filename")
            break
        else
            save(File{format"PNG"}(maskdir("$filename" * ".png")), mask)
        end
    end
end
